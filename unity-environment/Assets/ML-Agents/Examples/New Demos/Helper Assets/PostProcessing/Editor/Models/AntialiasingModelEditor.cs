using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Method = AntialiasingModel.Method;
    using Settings = AntialiasingModel.Settings;

    [PostProcessingModelEditor(typeof(AntialiasingModel))]
    public class AntialiasingModelEditor : PostProcessingModelEditor
    {
        SerializedProperty m_Method;

        SerializedProperty m_FxaaPreset;

        SerializedProperty m_TaaJitterSpread;
        SerializedProperty m_TaaSharpen;
        SerializedProperty m_TaaStationaryBlending;
        SerializedProperty m_TaaMotionBlending;

        static string[] s_MethodNames =
        {
            "Fast Approximate Anti-aliasing",
            "Temporal Anti-aliasing"
        };

        public override void OnEnable()
        {
            m_Method = FindSetting((Settings x) => x.method);

            m_FxaaPreset = FindSetting((Settings x) => x.fxaaSettings.preset);

            m_TaaJitterSpread = FindSetting((Settings x) => x.taaSettings.jitterSpread);
            m_TaaSharpen = FindSetting((Settings x) => x.taaSettings.sharpen);
            m_TaaStationaryBlending = FindSetting((Settings x) => x.taaSettings.stationaryBlending);
            m_TaaMotionBlending = FindSetting((Settings x) => x.taaSettings.motionBlending);
        }

        public override void OnInspectorGUI()
        {
            m_Method.intValue = EditorGUILayout.Popup("Method", m_Method.intValue, s_MethodNames);

            if (m_Method.intValue == (int)Method.Fxaa)
            {
                EditorGUILayout.PropertyField(m_FxaaPreset);
            }
            else if (m_Method.intValue == (int)Method.Taa)
            {
                if (QualitySettings.antiAliasing > 1)
                    EditorGUILayout.HelpBox("Temporal Anti-Aliasing doesn't work correctly when MSAA is enabled.", MessageType.Warning);

                EditorGUILayout.LabelField("Jitter", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_TaaJitterSpread, EditorGUIHelper.GetContent("Spread"));
                EditorGUI.indentLevel--;

                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Blending", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_TaaStationaryBlending, EditorGUIHelper.GetContent("Stationary"));
                EditorGUILayout.PropertyField(m_TaaMotionBlending, EditorGUIHelper.GetContent("Motion"));
                EditorGUI.indentLevel--;

                EditorGUILayout.Space();

                EditorGUILayout.PropertyField(m_TaaSharpen);
            }
        }
    }
}
