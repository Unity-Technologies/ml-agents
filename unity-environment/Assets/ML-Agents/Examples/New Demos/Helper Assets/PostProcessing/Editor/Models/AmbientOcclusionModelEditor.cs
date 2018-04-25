using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Settings = AmbientOcclusionModel.Settings;

    [PostProcessingModelEditor(typeof(AmbientOcclusionModel))]
    public class AmbientOcclusionModelEditor : PostProcessingModelEditor
    {
        SerializedProperty m_Intensity;
        SerializedProperty m_Radius;
        SerializedProperty m_SampleCount;
        SerializedProperty m_Downsampling;
        SerializedProperty m_ForceForwardCompatibility;
        SerializedProperty m_AmbientOnly;
        SerializedProperty m_HighPrecision;

        public override void OnEnable()
        {
            m_Intensity = FindSetting((Settings x) => x.intensity);
            m_Radius = FindSetting((Settings x) => x.radius);
            m_SampleCount = FindSetting((Settings x) => x.sampleCount);
            m_Downsampling = FindSetting((Settings x) => x.downsampling);
            m_ForceForwardCompatibility = FindSetting((Settings x) => x.forceForwardCompatibility);
            m_AmbientOnly = FindSetting((Settings x) => x.ambientOnly);
            m_HighPrecision = FindSetting((Settings x) => x.highPrecision);
        }

        public override void OnInspectorGUI()
        {
            EditorGUILayout.PropertyField(m_Intensity);
            EditorGUILayout.PropertyField(m_Radius);
            EditorGUILayout.PropertyField(m_SampleCount);
            EditorGUILayout.PropertyField(m_Downsampling);
            EditorGUILayout.PropertyField(m_ForceForwardCompatibility);
            EditorGUILayout.PropertyField(m_HighPrecision, EditorGUIHelper.GetContent("High Precision (Forward)"));

            using (new EditorGUI.DisabledGroupScope(m_ForceForwardCompatibility.boolValue))
                EditorGUILayout.PropertyField(m_AmbientOnly, EditorGUIHelper.GetContent("Ambient Only (Deferred + HDR)"));
        }
    }
}
