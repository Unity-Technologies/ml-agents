#if UNITY_EDITOR
using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Settings = EyeAdaptationModel.Settings;

    [PostProcessingModelEditor(typeof(EyeAdaptationModel))]
    public class EyeAdaptationModelEditor : PostProcessingModelEditor
    {
        SerializedProperty m_LowPercent;
        SerializedProperty m_HighPercent;
        SerializedProperty m_MinLuminance;
        SerializedProperty m_MaxLuminance;
        SerializedProperty m_KeyValue;
        SerializedProperty m_DynamicKeyValue;
        SerializedProperty m_AdaptationType;
        SerializedProperty m_SpeedUp;
        SerializedProperty m_SpeedDown;
        SerializedProperty m_LogMin;
        SerializedProperty m_LogMax;

        public override void OnEnable()
        {
            m_LowPercent = FindSetting((Settings x) => x.lowPercent);
            m_HighPercent = FindSetting((Settings x) => x.highPercent);
            m_MinLuminance = FindSetting((Settings x) => x.minLuminance);
            m_MaxLuminance = FindSetting((Settings x) => x.maxLuminance);
            m_KeyValue = FindSetting((Settings x) => x.keyValue);
            m_DynamicKeyValue = FindSetting((Settings x) => x.dynamicKeyValue);
            m_AdaptationType = FindSetting((Settings x) => x.adaptationType);
            m_SpeedUp = FindSetting((Settings x) => x.speedUp);
            m_SpeedDown = FindSetting((Settings x) => x.speedDown);
            m_LogMin = FindSetting((Settings x) => x.logMin);
            m_LogMax = FindSetting((Settings x) => x.logMax);
        }

        public override void OnInspectorGUI()
        {
            if (!GraphicsUtils.supportsDX11)
                EditorGUILayout.HelpBox("This effect requires support for compute shaders. Enabling it won't do anything on unsupported platforms.", MessageType.Warning);

            EditorGUILayout.LabelField("Luminosity range", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(m_LogMin, EditorGUIHelper.GetContent("Minimum (EV)"));
            EditorGUILayout.PropertyField(m_LogMax, EditorGUIHelper.GetContent("Maximum (EV)"));
            EditorGUI.indentLevel--;
            EditorGUILayout.Space();

            EditorGUILayout.LabelField("Auto exposure", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            float low = m_LowPercent.floatValue;
            float high = m_HighPercent.floatValue;

            EditorGUILayout.MinMaxSlider(EditorGUIHelper.GetContent("Histogram filtering|These values are the lower and upper percentages of the histogram that will be used to find a stable average luminance. Values outside of this range will be discarded and won't contribute to the average luminance."), ref low, ref high, 1f, 99f);

            m_LowPercent.floatValue = low;
            m_HighPercent.floatValue = high;

            EditorGUILayout.PropertyField(m_MinLuminance, EditorGUIHelper.GetContent("Minimum (EV)"));
            EditorGUILayout.PropertyField(m_MaxLuminance, EditorGUIHelper.GetContent("Maximum (EV)"));
            EditorGUILayout.PropertyField(m_DynamicKeyValue);

            if (!m_DynamicKeyValue.boolValue)
                EditorGUILayout.PropertyField(m_KeyValue);

            EditorGUI.indentLevel--;
            EditorGUILayout.Space();

            EditorGUILayout.LabelField("Adaptation", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;

            EditorGUILayout.PropertyField(m_AdaptationType, EditorGUIHelper.GetContent("Type"));

            if (m_AdaptationType.intValue == (int)EyeAdaptationModel.EyeAdaptationType.Progressive)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_SpeedUp);
                EditorGUILayout.PropertyField(m_SpeedDown);
                EditorGUI.indentLevel--;
            }

            EditorGUI.indentLevel--;
        }
    }
}
#endif
