namespace UnityEngine.PostProcessing
{
    public sealed class EyeAdaptationComponent : PostProcessingComponentRenderTexture<EyeAdaptationModel>
    {
        static class Uniforms
        {
            internal static readonly int _Params               = Shader.PropertyToID("_Params");
            internal static readonly int _Speed                = Shader.PropertyToID("_Speed");
            internal static readonly int _ScaleOffsetRes       = Shader.PropertyToID("_ScaleOffsetRes");
            internal static readonly int _ExposureCompensation = Shader.PropertyToID("_ExposureCompensation");
            internal static readonly int _AutoExposure         = Shader.PropertyToID("_AutoExposure");
            internal static readonly int _DebugWidth           = Shader.PropertyToID("_DebugWidth");
        }

        ComputeShader m_EyeCompute;
        ComputeBuffer m_HistogramBuffer;

        readonly RenderTexture[] m_AutoExposurePool = new RenderTexture[2];
        int m_AutoExposurePingPing;
        RenderTexture m_CurrentAutoExposure;

        RenderTexture m_DebugHistogram;

        static uint[] s_EmptyHistogramBuffer;

        bool m_FirstFrame = true;

        // Don't forget to update 'EyeAdaptation.cginc' if you change these values !
        const int k_HistogramBins = 64;
        const int k_HistogramThreadX = 16;
        const int k_HistogramThreadY = 16;

        public override bool active
        {
            get
            {
                return model.enabled
                       && SystemInfo.supportsComputeShaders
                       && !context.interrupted;
            }
        }

        public void ResetHistory()
        {
            m_FirstFrame = true;
        }

        public override void OnEnable()
        {
            m_FirstFrame = true;
        }

        public override void OnDisable()
        {
            foreach (var rt in m_AutoExposurePool)
                GraphicsUtils.Destroy(rt);

            if (m_HistogramBuffer != null)
                m_HistogramBuffer.Release();

            m_HistogramBuffer = null;

            if (m_DebugHistogram != null)
                m_DebugHistogram.Release();

            m_DebugHistogram = null;
        }

        Vector4 GetHistogramScaleOffsetRes()
        {
            var settings = model.settings;
            float diff = settings.logMax - settings.logMin;
            float scale = 1f / diff;
            float offset = -settings.logMin * scale;
            return new Vector4(scale, offset, Mathf.Floor(context.width / 2f), Mathf.Floor(context.height / 2f));
        }

        public Texture Prepare(RenderTexture source, Material uberMaterial)
        {
            var settings = model.settings;

            // Setup compute
            if (m_EyeCompute == null)
                m_EyeCompute = Resources.Load<ComputeShader>("Shaders/EyeHistogram");

            var material = context.materialFactory.Get("Hidden/Post FX/Eye Adaptation");
            material.shaderKeywords = null;

            if (m_HistogramBuffer == null)
                m_HistogramBuffer = new ComputeBuffer(k_HistogramBins, sizeof(uint));

            if (s_EmptyHistogramBuffer == null)
                s_EmptyHistogramBuffer = new uint[k_HistogramBins];

            // Downscale the framebuffer, we don't need an absolute precision for auto exposure and it
            // helps making it more stable
            var scaleOffsetRes = GetHistogramScaleOffsetRes();

            var rt = context.renderTextureFactory.Get((int)scaleOffsetRes.z, (int)scaleOffsetRes.w, 0, source.format);
            Graphics.Blit(source, rt);

            if (m_AutoExposurePool[0] == null || !m_AutoExposurePool[0].IsCreated())
                m_AutoExposurePool[0] = new RenderTexture(1, 1, 0, RenderTextureFormat.RFloat);

            if (m_AutoExposurePool[1] == null || !m_AutoExposurePool[1].IsCreated())
                m_AutoExposurePool[1] = new RenderTexture(1, 1, 0, RenderTextureFormat.RFloat);

            // Clears the buffer on every frame as we use it to accumulate luminance values on each frame
            m_HistogramBuffer.SetData(s_EmptyHistogramBuffer);

            // Gets a log histogram
            int kernel = m_EyeCompute.FindKernel("KEyeHistogram");
            m_EyeCompute.SetBuffer(kernel, "_Histogram", m_HistogramBuffer);
            m_EyeCompute.SetTexture(kernel, "_Source", rt);
            m_EyeCompute.SetVector("_ScaleOffsetRes", scaleOffsetRes);
            m_EyeCompute.Dispatch(kernel, Mathf.CeilToInt(rt.width / (float)k_HistogramThreadX), Mathf.CeilToInt(rt.height / (float)k_HistogramThreadY), 1);

            // Cleanup
            context.renderTextureFactory.Release(rt);

            // Make sure filtering values are correct to avoid apocalyptic consequences
            const float minDelta = 1e-2f;
            settings.highPercent = Mathf.Clamp(settings.highPercent, 1f + minDelta, 99f);
            settings.lowPercent = Mathf.Clamp(settings.lowPercent, 1f, settings.highPercent - minDelta);

            // Compute auto exposure
            material.SetBuffer("_Histogram", m_HistogramBuffer); // No (int, buffer) overload for SetBuffer ?
            material.SetVector(Uniforms._Params, new Vector4(settings.lowPercent * 0.01f, settings.highPercent * 0.01f, Mathf.Exp(settings.minLuminance * 0.69314718055994530941723212145818f), Mathf.Exp(settings.maxLuminance * 0.69314718055994530941723212145818f)));
            material.SetVector(Uniforms._Speed, new Vector2(settings.speedDown, settings.speedUp));
            material.SetVector(Uniforms._ScaleOffsetRes, scaleOffsetRes);
            material.SetFloat(Uniforms._ExposureCompensation, settings.keyValue);

            if (settings.dynamicKeyValue)
                material.EnableKeyword("AUTO_KEY_VALUE");

            if (m_FirstFrame || !Application.isPlaying)
            {
                // We don't want eye adaptation when not in play mode because the GameView isn't
                // animated, thus making it harder to tweak. Just use the final audo exposure value.
                m_CurrentAutoExposure = m_AutoExposurePool[0];
                Graphics.Blit(null, m_CurrentAutoExposure, material, (int)EyeAdaptationModel.EyeAdaptationType.Fixed);

                // Copy current exposure to the other pingpong target to avoid adapting from black
                Graphics.Blit(m_AutoExposurePool[0], m_AutoExposurePool[1]);
            }
            else
            {
                int pp = m_AutoExposurePingPing;
                var src = m_AutoExposurePool[++pp % 2];
                var dst = m_AutoExposurePool[++pp % 2];
                Graphics.Blit(src, dst, material, (int)settings.adaptationType);
                m_AutoExposurePingPing = ++pp % 2;
                m_CurrentAutoExposure = dst;
            }

            // Generate debug histogram
            if (context.profile.debugViews.IsModeActive(BuiltinDebugViewsModel.Mode.EyeAdaptation))
            {
                if (m_DebugHistogram == null || !m_DebugHistogram.IsCreated())
                {
                    m_DebugHistogram = new RenderTexture(256, 128, 0, RenderTextureFormat.ARGB32)
                    {
                        filterMode = FilterMode.Point,
                        wrapMode = TextureWrapMode.Clamp
                    };
                }

                material.SetFloat(Uniforms._DebugWidth, m_DebugHistogram.width);
                Graphics.Blit(null, m_DebugHistogram, material, 2);
            }

            m_FirstFrame = false;
            return m_CurrentAutoExposure;
        }

        public void OnGUI()
        {
            if (m_DebugHistogram == null || !m_DebugHistogram.IsCreated())
                return;

            var rect = new Rect(context.viewport.x * Screen.width + 8f, 8f, m_DebugHistogram.width, m_DebugHistogram.height);
            GUI.DrawTexture(rect, m_DebugHistogram);
        }
    }
}
