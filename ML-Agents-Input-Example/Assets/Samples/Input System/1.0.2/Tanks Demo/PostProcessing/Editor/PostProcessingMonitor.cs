#if UNITY_EDITOR
using System;
using UnityEngine;

namespace UnityEditor.PostProcessing
{
    using MonitorSettings = UnityEngine.PostProcessing.PostProcessingProfile.MonitorSettings;

    public abstract class PostProcessingMonitor : IDisposable
    {
        protected MonitorSettings m_MonitorSettings;
        protected PostProcessingInspector m_BaseEditor;

        public void Init(MonitorSettings monitorSettings, PostProcessingInspector baseEditor)
        {
            m_MonitorSettings = monitorSettings;
            m_BaseEditor = baseEditor;
        }

        public abstract bool IsSupported();

        public abstract GUIContent GetMonitorTitle();

        public virtual void OnMonitorSettings()
        {}

        public abstract void OnMonitorGUI(Rect r);

        public virtual void OnFrameData(RenderTexture source)
        {}

        public virtual void Dispose()
        {}
    }
}
#endif
