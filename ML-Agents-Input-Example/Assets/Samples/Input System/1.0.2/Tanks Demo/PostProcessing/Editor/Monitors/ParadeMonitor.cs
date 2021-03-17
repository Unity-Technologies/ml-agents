#if UNITY_EDITOR
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    public class ParadeMonitor : PostProcessingMonitor
    {
        static GUIContent s_MonitorTitle = new GUIContent("Parade");

        ComputeShader m_ComputeShader;
        ComputeBuffer m_Buffer;
        Material m_Material;
        RenderTexture m_WaveformTexture;
        Rect m_MonitorAreaRect;

        public ParadeMonitor()
        {
            m_ComputeShader = EditorResources.Load<ComputeShader>("Monitors/WaveformCompute.compute");
        }

        public override void Dispose()
        {
            GraphicsUtils.Destroy(m_Material);
            GraphicsUtils.Destroy(m_WaveformTexture);

            if (m_Buffer != null)
                m_Buffer.Release();

            m_Material = null;
            m_WaveformTexture = null;
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
            float exposure = m_MonitorSettings.paradeExposure;

            refreshOnPlay = GUILayout.Toggle(refreshOnPlay, new GUIContent(FxStyles.playIcon, "Keep refreshing the parade in play mode; this may impact performances."), FxStyles.preButton);
            exposure = GUILayout.HorizontalSlider(exposure, 0.05f, 0.3f, FxStyles.preSlider, FxStyles.preSliderThumb, GUILayout.Width(40f));

            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(m_BaseEditor.serializedObject.targetObject, "Parade Settings Changed");
                m_MonitorSettings.refreshOnPlay = refreshOnPlay;
                m_MonitorSettings.paradeExposure = exposure;
                InternalEditorUtility.RepaintAllViews();
            }
        }

        public override void OnMonitorGUI(Rect r)
        {
            if (Event.current.type == EventType.Repaint)
            {
                // If m_MonitorAreaRect isn't set the preview was just opened so refresh the render to get the waveform data
                if (Mathf.Approximately(m_MonitorAreaRect.width, 0) && Mathf.Approximately(m_MonitorAreaRect.height, 0))
                    InternalEditorUtility.RepaintAllViews();

                // Sizing
                float width = m_WaveformTexture != null
                    ? Mathf.Min(m_WaveformTexture.width, r.width - 65f)
                    : r.width;
                float height = m_WaveformTexture != null
                    ? Mathf.Min(m_WaveformTexture.height, r.height - 45f)
                    : r.height;

                m_MonitorAreaRect = new Rect(
                    Mathf.Floor(r.x + r.width / 2f - width / 2f),
                    Mathf.Floor(r.y + r.height / 2f - height / 2f - 5f),
                    width, height
                );

                if (m_WaveformTexture != null)
                {
                    m_Material.SetFloat("_Exposure", m_MonitorSettings.paradeExposure);

                    var oldActive = RenderTexture.active;
                    Graphics.Blit(null, m_WaveformTexture, m_Material, 0);
                    RenderTexture.active = oldActive;

                    Graphics.DrawTexture(m_MonitorAreaRect, m_WaveformTexture);

                    var color = Color.white;
                    const float kTickSize = 5f;

                    // Rect, lines & ticks points
                    //  A O B P C Q D
                    //  N           E
                    //  M           F
                    //  L           G
                    //  K T J S I R H

                    var A = new Vector3(m_MonitorAreaRect.x, m_MonitorAreaRect.y);
                    var D = new Vector3(A.x + m_MonitorAreaRect.width + 1f, m_MonitorAreaRect.y);
                    var H = new Vector3(D.x, D.y + m_MonitorAreaRect.height + 1f);
                    var K = new Vector3(A.x, H.y);

                    var F = new Vector3(D.x, D.y + (H.y - D.y) / 2f);
                    var M = new Vector3(A.x, A.y + (K.y - A.y) / 2f);

                    var B = new Vector3(A.x + (D.x - A.x) / 3f, A.y);
                    var C = new Vector3(A.x + (D.x - A.x) * 2f / 3f, A.y);
                    var I = new Vector3(K.x + (H.x - K.x) * 2f / 3f, K.y);
                    var J = new Vector3(K.x + (H.x - K.x) / 3f, K.y);

                    var N = new Vector3(A.x, A.y + (M.y - A.y) / 2f);
                    var L = new Vector3(A.x, M.y + (K.y - M.y) / 2f);
                    var E = new Vector3(D.x, D.y + (F.y - D.y) / 2f);
                    var G = new Vector3(D.x, F.y + (H.y - F.y) / 2f);

                    var O = new Vector3(A.x + (B.x - A.x) / 2f, A.y);
                    var P = new Vector3(B.x + (C.x - B.x) / 2f, B.y);
                    var Q = new Vector3(C.x + (D.x - C.x) / 2f, C.y);

                    var R = new Vector3(I.x + (H.x - I.x) / 2f, I.y);
                    var S = new Vector3(J.x + (I.x - J.x) / 2f, J.y);
                    var T = new Vector3(K.x + (J.x - K.x) / 2f, K.y);

                    // Borders
                    Handles.color = color;
                    Handles.DrawLine(A, D);
                    Handles.DrawLine(D, H);
                    Handles.DrawLine(H, K);
                    Handles.DrawLine(K, new Vector3(A.x, A.y - 1f));

                    Handles.DrawLine(B, J);
                    Handles.DrawLine(C, I);

                    // Vertical ticks
                    Handles.DrawLine(A, new Vector3(A.x - kTickSize, A.y));
                    Handles.DrawLine(N, new Vector3(N.x - kTickSize, N.y));
                    Handles.DrawLine(M, new Vector3(M.x - kTickSize, M.y));
                    Handles.DrawLine(L, new Vector3(L.x - kTickSize, L.y));
                    Handles.DrawLine(K, new Vector3(K.x - kTickSize, K.y));

                    Handles.DrawLine(D, new Vector3(D.x + kTickSize, D.y));
                    Handles.DrawLine(E, new Vector3(E.x + kTickSize, E.y));
                    Handles.DrawLine(F, new Vector3(F.x + kTickSize, F.y));
                    Handles.DrawLine(G, new Vector3(G.x + kTickSize, G.y));
                    Handles.DrawLine(H, new Vector3(H.x + kTickSize, H.y));

                    // Horizontal ticks
                    Handles.DrawLine(A, new Vector3(A.x, A.y - kTickSize));
                    Handles.DrawLine(B, new Vector3(B.x, B.y - kTickSize));
                    Handles.DrawLine(C, new Vector3(C.x, C.y - kTickSize));
                    Handles.DrawLine(D, new Vector3(D.x, D.y - kTickSize));
                    Handles.DrawLine(O, new Vector3(O.x, O.y - kTickSize));
                    Handles.DrawLine(P, new Vector3(P.x, P.y - kTickSize));
                    Handles.DrawLine(Q, new Vector3(Q.x, Q.y - kTickSize));

                    Handles.DrawLine(H, new Vector3(H.x, H.y + kTickSize));
                    Handles.DrawLine(I, new Vector3(I.x, I.y + kTickSize));
                    Handles.DrawLine(J, new Vector3(J.x, J.y + kTickSize));
                    Handles.DrawLine(K, new Vector3(K.x, K.y + kTickSize));
                    Handles.DrawLine(R, new Vector3(R.x, R.y + kTickSize));
                    Handles.DrawLine(S, new Vector3(S.x, S.y + kTickSize));
                    Handles.DrawLine(T, new Vector3(T.x, T.y + kTickSize));

                    // Labels
                    GUI.color = color;
                    GUI.Label(new Rect(A.x - kTickSize - 34f, A.y - 15f, 30f, 30f), "1.0", FxStyles.tickStyleRight);
                    GUI.Label(new Rect(M.x - kTickSize - 34f, M.y - 15f, 30f, 30f), "0.5", FxStyles.tickStyleRight);
                    GUI.Label(new Rect(K.x - kTickSize - 34f, K.y - 15f, 30f, 30f), "0.0", FxStyles.tickStyleRight);

                    GUI.Label(new Rect(D.x + kTickSize + 4f, D.y - 15f, 30f, 30f), "1.0", FxStyles.tickStyleLeft);
                    GUI.Label(new Rect(F.x + kTickSize + 4f, F.y - 15f, 30f, 30f), "0.5", FxStyles.tickStyleLeft);
                    GUI.Label(new Rect(H.x + kTickSize + 4f, H.y - 15f, 30f, 30f), "0.0", FxStyles.tickStyleLeft);
                }
            }
        }

        public override void OnFrameData(RenderTexture source)
        {
            if (Application.isPlaying && !m_MonitorSettings.refreshOnPlay)
                return;

            if (Mathf.Approximately(m_MonitorAreaRect.width, 0) || Mathf.Approximately(m_MonitorAreaRect.height, 0))
                return;

            float ratio = ((float)source.width / (float)source.height) / 3f;
            int h = 384;
            int w = Mathf.FloorToInt(h * ratio);

            var rt = RenderTexture.GetTemporary(w, h, 0, source.format);
            Graphics.Blit(source, rt);
            ComputeWaveform(rt);
            m_BaseEditor.Repaint();
            RenderTexture.ReleaseTemporary(rt);
        }

        void CreateBuffer(int width, int height)
        {
            m_Buffer = new ComputeBuffer(width * height, sizeof(uint) << 2);
        }

        void ComputeWaveform(RenderTexture source)
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

            var channels = m_MonitorSettings.waveformY
                ? new Vector4(0f, 0f, 0f, 1f)
                : new Vector4(m_MonitorSettings.waveformR ? 1f : 0f, m_MonitorSettings.waveformG ? 1f : 0f, m_MonitorSettings.waveformB ? 1f : 0f, 0f);

            var cs = m_ComputeShader;

            int kernel = cs.FindKernel("KWaveformClear");
            cs.SetBuffer(kernel, "_Waveform", m_Buffer);
            cs.Dispatch(kernel, source.width, 1, 1);

            kernel = cs.FindKernel("KWaveform");
            cs.SetBuffer(kernel, "_Waveform", m_Buffer);
            cs.SetTexture(kernel, "_Source", source);
            cs.SetInt("_IsLinear", GraphicsUtils.isLinearColorSpace ? 1 : 0);
            cs.SetVector("_Channels", channels);
            cs.Dispatch(kernel, source.width, 1, 1);

            if (m_WaveformTexture == null || m_WaveformTexture.width != (source.width * 3) || m_WaveformTexture.height != source.height)
            {
                GraphicsUtils.Destroy(m_WaveformTexture);
                m_WaveformTexture = new RenderTexture(source.width * 3, source.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
                {
                    hideFlags = HideFlags.DontSave,
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
            }

            if (m_Material == null)
                m_Material = new Material(Shader.Find("Hidden/Post FX/Monitors/Parade Render")) { hideFlags = HideFlags.DontSave };

            m_Material.SetBuffer("_Waveform", m_Buffer);
            m_Material.SetVector("_Size", new Vector2(m_WaveformTexture.width, m_WaveformTexture.height));
            m_Material.SetVector("_Channels", channels);
        }
    }
}
#endif
