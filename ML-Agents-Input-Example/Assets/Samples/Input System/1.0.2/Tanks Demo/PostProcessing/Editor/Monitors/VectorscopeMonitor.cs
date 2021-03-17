#if UNITY_EDITOR
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    public class VectorscopeMonitor : PostProcessingMonitor
    {
        static GUIContent s_MonitorTitle = new GUIContent("Vectorscope");

        ComputeShader m_ComputeShader;
        ComputeBuffer m_Buffer;
        Material m_Material;
        RenderTexture m_VectorscopeTexture;
        Rect m_MonitorAreaRect;

        public VectorscopeMonitor()
        {
            m_ComputeShader = EditorResources.Load<ComputeShader>("Monitors/VectorscopeCompute.compute");
        }

        public override void Dispose()
        {
            GraphicsUtils.Destroy(m_Material);
            GraphicsUtils.Destroy(m_VectorscopeTexture);

            if (m_Buffer != null)
                m_Buffer.Release();

            m_Material = null;
            m_VectorscopeTexture = null;
            m_Buffer = null;
        }

        public override bool IsSupported()
        {
            return m_ComputeShader != null && GraphicsUtils.supportsDX11;
        }

        public override GUIContent GetMonitorTitle()
        {
            return s_MonitorTitle;
        }

        public override void OnMonitorSettings()
        {
            EditorGUI.BeginChangeCheck();

            bool refreshOnPlay = m_MonitorSettings.refreshOnPlay;
            float exposure = m_MonitorSettings.vectorscopeExposure;
            bool showBackground = m_MonitorSettings.vectorscopeShowBackground;

            refreshOnPlay = GUILayout.Toggle(refreshOnPlay, new GUIContent(FxStyles.playIcon, "Keep refreshing the vectorscope in play mode; this may impact performances."), FxStyles.preButton);
            exposure = GUILayout.HorizontalSlider(exposure, 0.05f, 0.3f, FxStyles.preSlider, FxStyles.preSliderThumb, GUILayout.Width(40f));
            showBackground = GUILayout.Toggle(showBackground, new GUIContent(FxStyles.checkerIcon, "Show an YUV background in the vectorscope."), FxStyles.preButton);

            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(m_BaseEditor.serializedObject.targetObject, "Vectorscope Settings Changed");
                m_MonitorSettings.refreshOnPlay = refreshOnPlay;
                m_MonitorSettings.vectorscopeExposure = exposure;
                m_MonitorSettings.vectorscopeShowBackground = showBackground;
                InternalEditorUtility.RepaintAllViews();
            }
        }

        public override void OnMonitorGUI(Rect r)
        {
            if (Event.current.type == EventType.Repaint)
            {
                // If m_MonitorAreaRect isn't set the preview was just opened so refresh the render to get the vectoscope data
                if (Mathf.Approximately(m_MonitorAreaRect.width, 0) && Mathf.Approximately(m_MonitorAreaRect.height, 0))
                    InternalEditorUtility.RepaintAllViews();

                // Sizing
                float size = 0f;

                if (r.width < r.height)
                {
                    size = m_VectorscopeTexture != null
                        ? Mathf.Min(m_VectorscopeTexture.width, r.width - 35f)
                        : r.width;
                }
                else
                {
                    size = m_VectorscopeTexture != null
                        ? Mathf.Min(m_VectorscopeTexture.height, r.height - 25f)
                        : r.height;
                }

                m_MonitorAreaRect = new Rect(
                    Mathf.Floor(r.x + r.width / 2f - size / 2f),
                    Mathf.Floor(r.y + r.height / 2f - size / 2f - 5f),
                    size, size
                );

                if (m_VectorscopeTexture != null)
                {
                    m_Material.SetFloat("_Exposure", m_MonitorSettings.vectorscopeExposure);

                    var oldActive = RenderTexture.active;
                    Graphics.Blit(null, m_VectorscopeTexture, m_Material, m_MonitorSettings.vectorscopeShowBackground ? 0 : 1);
                    RenderTexture.active = oldActive;

                    Graphics.DrawTexture(m_MonitorAreaRect, m_VectorscopeTexture);

                    var color = Color.white;
                    const float kTickSize = 10f;
                    const int kTickCount = 24;

                    float radius = m_MonitorAreaRect.width / 2f;
                    float midX = m_MonitorAreaRect.x + radius;
                    float midY = m_MonitorAreaRect.y + radius;
                    var center = new Vector2(midX, midY);

                    // Cross
                    color.a *= 0.5f;
                    Handles.color = color;
                    Handles.DrawLine(new Vector2(midX, m_MonitorAreaRect.y), new Vector2(midX, m_MonitorAreaRect.y + m_MonitorAreaRect.height));
                    Handles.DrawLine(new Vector2(m_MonitorAreaRect.x, midY), new Vector2(m_MonitorAreaRect.x + m_MonitorAreaRect.width, midY));

                    if (m_MonitorAreaRect.width > 100f)
                    {
                        color.a = 1f;

                        // Ticks
                        Handles.color = color;
                        for (int i = 0; i < kTickCount; i++)
                        {
                            float a = (float)i / (float)kTickCount;
                            float theta = a * (Mathf.PI * 2f);
                            float tx = Mathf.Cos(theta + (Mathf.PI / 2f));
                            float ty = Mathf.Sin(theta - (Mathf.PI / 2f));
                            var innerVec = center + new Vector2(tx, ty) * (radius - kTickSize);
                            var outerVec = center + new Vector2(tx, ty) * radius;
                            Handles.DrawAAPolyLine(3f, innerVec, outerVec);
                        }

                        // Labels (where saturation reaches 75%)
                        color.a = 1f;
                        var oldColor = GUI.color;
                        GUI.color = color * 2f;

                        var point = new Vector2(-0.254f, -0.750f) * radius + center;
                        var rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[R]", FxStyles.tickStyleCenter);

                        point = new Vector2(-0.497f, 0.629f) * radius + center;
                        rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[G]", FxStyles.tickStyleCenter);

                        point = new Vector2(0.750f, 0.122f) * radius + center;
                        rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[B]", FxStyles.tickStyleCenter);

                        point = new Vector2(-0.750f, -0.122f) * radius + center;
                        rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[Y]", FxStyles.tickStyleCenter);

                        point = new Vector2(0.254f, 0.750f) * radius + center;
                        rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[C]", FxStyles.tickStyleCenter);

                        point = new Vector2(0.497f, -0.629f) * radius + center;
                        rect = new Rect(point.x - 10f, point.y - 10f, 20f, 20f);
                        GUI.Label(rect, "[M]", FxStyles.tickStyleCenter);
                        GUI.color = oldColor;
                    }
                }
            }
        }

        public override void OnFrameData(RenderTexture source)
        {
            if (Application.isPlaying && !m_MonitorSettings.refreshOnPlay)
                return;

            if (Mathf.Approximately(m_MonitorAreaRect.width, 0) || Mathf.Approximately(m_MonitorAreaRect.height, 0))
                return;

            float ratio = (float)source.width / (float)source.height;
            int h = 384;
            int w = Mathf.FloorToInt(h * ratio);

            var rt = RenderTexture.GetTemporary(w, h, 0, source.format);
            Graphics.Blit(source, rt);
            ComputeVectorscope(rt);
            m_BaseEditor.Repaint();
            RenderTexture.ReleaseTemporary(rt);
        }

        void CreateBuffer(int width, int height)
        {
            m_Buffer = new ComputeBuffer(width * height, sizeof(uint));
        }

        void ComputeVectorscope(RenderTexture source)
        {
            if (m_Buffer == null)
            {
                CreateBuffer(source.width, source.height);
            }
            else if (m_Buffer.count != (source.width * source.height))
            {
                m_Buffer.Release();
                CreateBuffer(source.width, source.height);
            }

            var cs = m_ComputeShader;

            int kernel = cs.FindKernel("KVectorscopeClear");
            cs.SetBuffer(kernel, "_Vectorscope", m_Buffer);
            cs.SetVector("_Res", new Vector4(source.width, source.height, 0f, 0f));
            cs.Dispatch(kernel, Mathf.CeilToInt(source.width / 32f), Mathf.CeilToInt(source.height / 32f), 1);

            kernel = cs.FindKernel("KVectorscope");
            cs.SetBuffer(kernel, "_Vectorscope", m_Buffer);
            cs.SetTexture(kernel, "_Source", source);
            cs.SetInt("_IsLinear", GraphicsUtils.isLinearColorSpace ? 1 : 0);
            cs.SetVector("_Res", new Vector4(source.width, source.height, 0f, 0f));
            cs.Dispatch(kernel, Mathf.CeilToInt(source.width / 32f), Mathf.CeilToInt(source.height / 32f), 1);

            if (m_VectorscopeTexture == null || m_VectorscopeTexture.width != source.width || m_VectorscopeTexture.height != source.height)
            {
                GraphicsUtils.Destroy(m_VectorscopeTexture);
                m_VectorscopeTexture = new RenderTexture(source.width, source.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
                {
                    hideFlags = HideFlags.DontSave,
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
            }

            if (m_Material == null)
                m_Material = new Material(Shader.Find("Hidden/Post FX/Monitors/Vectorscope Render")) { hideFlags = HideFlags.DontSave };

            m_Material.SetBuffer("_Vectorscope", m_Buffer);
            m_Material.SetVector("_Size", new Vector2(m_VectorscopeTexture.width, m_VectorscopeTexture.height));
        }
    }
}
#endif
