using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Mode = BuiltinDebugViewsModel.Mode;
    using Settings = BuiltinDebugViewsModel.Settings;

    [PostProcessingModelEditor(typeof(BuiltinDebugViewsModel), alwaysEnabled: true)]
    public class BuiltinDebugViewsEditor : PostProcessingModelEditor
    {
        struct DepthSettings
        {
            public SerializedProperty scale;
        }

        struct MotionVectorsSettings
        {
            public SerializedProperty sourceOpacity;
            public SerializedProperty motionImageOpacity;
            public SerializedProperty motionImageAmplitude;
            public SerializedProperty motionVectorsOpacity;
            public SerializedProperty motionVectorsResolution;
            public SerializedProperty motionVectorsAmplitude;
        }

        SerializedProperty m_Mode;
        DepthSettings m_Depth;
        MotionVectorsSettings m_MotionVectors;

        public override void OnEnable()
        {
            m_Mode = FindSetting((Settings x) => x.mode);

            m_Depth = new DepthSettings
            {
                scale = FindSetting((Settings x) => x.depth.scale)
            };

            m_MotionVectors = new MotionVectorsSettings
            {
                sourceOpacity = FindSetting((Settings x) => x.motionVectors.sourceOpacity),
                motionImageOpacity = FindSetting((Settings x) => x.motionVectors.motionImageOpacity),
                motionImageAmplitude = FindSetting((Settings x) => x.motionVectors.motionImageAmplitude),
                motionVectorsOpacity = FindSetting((Settings x) => x.motionVectors.motionVectorsOpacity),
                motionVectorsResolution = FindSetting((Settings x) => x.motionVectors.motionVectorsResolution),
                motionVectorsAmplitude = FindSetting((Settings x) => x.motionVectors.motionVectorsAmplitude),
            };
        }

        public override void OnInspectorGUI()
        {
            EditorGUILayout.PropertyField(m_Mode);

            int mode = m_Mode.intValue;

            if (mode == (int)Mode.Depth)
            {
                EditorGUILayout.PropertyField(m_Depth.scale);
            }
            else if (mode == (int)Mode.MotionVectors)
            {
                EditorGUILayout.HelpBox("Switch to play mode to see motion vectors.", MessageType.Info);

                EditorGUILayout.LabelField("Source Image", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_MotionVectors.sourceOpacity, EditorGUIHelper.GetContent("Opacity"));
                EditorGUI.indentLevel--;

                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Motion Vectors (overlay)", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;

                if (m_MotionVectors.motionImageOpacity.floatValue > 0f)
                    EditorGUILayout.HelpBox("Please keep opacity to 0 if you're subject to motion sickness.", MessageType.Warning);

                EditorGUILayout.PropertyField(m_MotionVectors.motionImageOpacity, EditorGUIHelper.GetContent("Opacity"));
                EditorGUILayout.PropertyField(m_MotionVectors.motionImageAmplitude, EditorGUIHelper.GetContent("Amplitude"));
                EditorGUI.indentLevel--;

                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Motion Vectors (arrows)", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_MotionVectors.motionVectorsOpacity, EditorGUIHelper.GetContent("Opacity"));
                EditorGUILayout.PropertyField(m_MotionVectors.motionVectorsResolution, EditorGUIHelper.GetContent("Resolution"));
                EditorGUILayout.PropertyField(m_MotionVectors.motionVectorsAmplitude, EditorGUIHelper.GetContent("Amplitude"));
                EditorGUI.indentLevel--;
            }
            else
            {
                CheckActiveEffect(mode == (int)Mode.AmbientOcclusion && !profile.ambientOcclusion.enabled, "Ambient Occlusion");
                CheckActiveEffect(mode == (int)Mode.FocusPlane && !profile.depthOfField.enabled, "Depth Of Field");
                CheckActiveEffect(mode == (int)Mode.EyeAdaptation && !profile.eyeAdaptation.enabled, "Eye Adaptation");
                CheckActiveEffect((mode == (int)Mode.LogLut || mode == (int)Mode.PreGradingLog) && !profile.colorGrading.enabled, "Color Grading");
                CheckActiveEffect(mode == (int)Mode.UserLut && !profile.userLut.enabled, "User Lut");
            }
        }

        void CheckActiveEffect(bool expr, string name)
        {
            if (expr)
                EditorGUILayout.HelpBox(string.Format("{0} isn't enabled, the debug view won't work.", name), MessageType.Warning);
        }
    }
}
