#if UNITY_EDITOR
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Settings = DepthOfFieldModel.Settings;

    [PostProcessingModelEditor(typeof(DepthOfFieldModel))]
    public class DepthOfFieldModelEditor : PostProcessingModelEditor
    {
        SerializedProperty m_FocusDistance;
        SerializedProperty m_Aperture;
        SerializedProperty m_FocalLength;
        SerializedProperty m_UseCameraFov;
        SerializedProperty m_KernelSize;

        public override void OnEnable()
        {
            m_FocusDistance = FindSetting((Settings x) => x.focusDistance);
            m_Aperture = FindSetting((Settings x) => x.aperture);
            m_FocalLength = FindSetting((Settings x) => x.focalLength);
            m_UseCameraFov = FindSetting((Settings x) => x.useCameraFov);
            m_KernelSize = FindSetting((Settings x) => x.kernelSize);
        }

        public override void OnInspectorGUI()
        {
            EditorGUILayout.PropertyField(m_FocusDistance);
            EditorGUILayout.PropertyField(m_Aperture, EditorGUIHelper.GetContent("Aperture (f-stop)"));

            EditorGUILayout.PropertyField(m_UseCameraFov, EditorGUIHelper.GetContent("Use Camera FOV"));
            if (!m_UseCameraFov.boolValue)
                EditorGUILayout.PropertyField(m_FocalLength, EditorGUIHelper.GetContent("Focal Length (mm)"));

            EditorGUILayout.PropertyField(m_KernelSize);
        }
    }
}
#endif
