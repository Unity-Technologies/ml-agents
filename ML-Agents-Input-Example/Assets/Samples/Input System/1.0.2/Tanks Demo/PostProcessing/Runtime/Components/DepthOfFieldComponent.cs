using UnityEngine.Rendering;

namespace UnityEngine.PostProcessing
{
    using DebugMode = BuiltinDebugViewsModel.Mode;

    public sealed class DepthOfFieldComponent : PostProcessingComponentRenderTexture<DepthOfFieldModel>
    {
        static class Uniforms
        {
            internal static readonly int _DepthOfFieldTex = Shader.PropertyToID("_DepthOfFieldTex");
            internal static readonly int _Distance = Shader.PropertyToID("_Distance");
            internal static readonly int _LensCoeff = Shader.PropertyToID("_LensCoeff");
            internal static readonly int _MaxCoC = Shader.PropertyToID("_MaxCoC");
            internal static readonly int _RcpMaxCoC = Shader.PropertyToID("_RcpMaxCoC");
            internal static readonly int _RcpAspect = Shader.PropertyToID("_RcpAspect");
            internal static readonly int _MainTex = Shader.PropertyToID("_MainTex");
            internal static readonly int _HistoryCoC = Shader.PropertyToID("_HistoryCoC");
            internal static readonly int _HistoryWeight = Shader.PropertyToID("_HistoryWeight");
            internal static readonly int _DepthOfFieldParams = Shader.PropertyToID("_DepthOfFieldParams");
        }

        const string k_ShaderString = "Hidden/Post FX/Depth Of Field";

        public override bool active
        {
            get
            {
                return model.enabled
                    && SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBHalf)
                    && SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RHalf)
                    && !context.interrupted;
            }
        }

        public override DepthTextureMode GetCameraFlags()
        {
            return DepthTextureMode.Depth;
        }

        RenderTexture m_CoCHistory;
        RenderBuffer[] m_MRT = new RenderBuffer[2];

        // Height of the 35mm full-frame format (36mm x 24mm)
        const float k_FilmHeight = 0.024f;

        float CalculateFocalLength()
        {
            var settings = model.settings;

            if (!settings.useCameraFov)
                return settings.focalLength / 1000f;

            float fov = context.camera.fieldOfView * Mathf.Deg2Rad;
            return 0.5f * k_FilmHeight / Mathf.Tan(0.5f * fov);
        }

        float CalculateMaxCoCRadius(int screenHeight)
        {
            // Estimate the allowable maximum radius of CoC from the kernel
            // size (the equation below was empirically derived).
            float radiusInPixels = (float)model.settings.kernelSize * 4f + 6f;

            // Applying a 5% limit to the CoC radius to keep the size of
            // TileMax/NeighborMax small enough.
            return Mathf.Min(0.05f, radiusInPixels / screenHeight);
        }

        public void Prepare(RenderTexture source, Material uberMaterial, bool antialiasCoC)
        {
            var settings = model.settings;

            // Material setup
            var material = context.materialFactory.Get(k_ShaderString);
            material.shaderKeywords = null;

            var s1 = settings.focusDistance;
            var f = CalculateFocalLength();
            s1 = Mathf.Max(s1, f);
            material.SetFloat(Uniforms._Distance, s1);

            var coeff = f * f / (settings.aperture * (s1 - f) * k_FilmHeight * 2);
            material.SetFloat(Uniforms._LensCoeff, coeff);

            var maxCoC = CalculateMaxCoCRadius(source.height);
            material.SetFloat(Uniforms._MaxCoC, maxCoC);
            material.SetFloat(Uniforms._RcpMaxCoC, 1f / maxCoC);

            var rcpAspect = (float)source.height / source.width;
            material.SetFloat(Uniforms._RcpAspect, rcpAspect);

            var rt1 = context.renderTextureFactory.Get(context.width / 2, context.height / 2, 0, RenderTextureFormat.ARGBHalf);
            source.filterMode = FilterMode.Point;

            // Pass #1 - Downsampling, prefiltering and CoC calculation
            if (!antialiasCoC)
            {
                Graphics.Blit(source, rt1, material, 0);
            }
            else
            {
                var initial = m_CoCHistory == null || !m_CoCHistory.IsCreated() || m_CoCHistory.width != context.width / 2 || m_CoCHistory.height != context.height / 2;

                var tempCoCHistory = RenderTexture.GetTemporary(context.width / 2, context.height / 2, 0, RenderTextureFormat.RHalf);
                tempCoCHistory.filterMode = FilterMode.Point;
                tempCoCHistory.name = "CoC History";

                m_MRT[0] = rt1.colorBuffer;
                m_MRT[1] = tempCoCHistory.colorBuffer;
                material.SetTexture(Uniforms._MainTex, source);
                material.SetTexture(Uniforms._HistoryCoC, m_CoCHistory);
                material.SetFloat(Uniforms._HistoryWeight, initial ? 0 : 0.5f);
                Graphics.SetRenderTarget(m_MRT, rt1.depthBuffer);
                GraphicsUtils.Blit(material, 1);

                RenderTexture.ReleaseTemporary(m_CoCHistory);
                m_CoCHistory = tempCoCHistory;
            }

            // Pass #2 - Bokeh simulation
            var rt2 = context.renderTextureFactory.Get(context.width / 2, context.height / 2, 0, RenderTextureFormat.ARGBHalf);
            Graphics.Blit(rt1, rt2, material, 2 + (int)settings.kernelSize);

            // Pass #3 - Postfilter blur
            Graphics.Blit(rt2, rt1, material, 6);

            if (context.profile.debugViews.IsModeActive(DebugMode.FocusPlane))
            {
                uberMaterial.SetVector(Uniforms._DepthOfFieldParams, new Vector2(s1, coeff));
                uberMaterial.EnableKeyword("DEPTH_OF_FIELD_COC_VIEW");
                context.Interrupt();
            }
            else
            {
                uberMaterial.SetTexture(Uniforms._DepthOfFieldTex, rt1);
                uberMaterial.EnableKeyword("DEPTH_OF_FIELD");
            }

            context.renderTextureFactory.Release(rt2);
            source.filterMode = FilterMode.Bilinear;
        }

        public override void OnDisable()
        {
            if (m_CoCHistory != null)
                RenderTexture.ReleaseTemporary(m_CoCHistory);

            m_CoCHistory = null;
        }
    }
}
