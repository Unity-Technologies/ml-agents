using UnityEditor;
using Unity.MLAgents.Integrations.Match3;
namespace Unity.MLAgents.Editor
{
    [CustomEditor(typeof(Match3SensorComponent), editorForChildClasses: true)]
    [CanEditMultipleObjects]
    internal class Match3SensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            var component = (Match3SensorComponent)target;
            var board = component.GetComponent<AbstractBoard>();
            if (board == null)
            {
                EditorGUILayout.HelpBox("You must provide an implementation of an AbstractBoard.", MessageType.Warning);
                return;
            }

            // Drawing the RenderTextureComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_ObservationType"), true);
            }
            EditorGUI.EndDisabledGroup();

            var requireSensorUpdate = EditorGUI.EndChangeCheck();
            so.ApplyModifiedProperties();

            if (requireSensorUpdate)
            {
                UpdateSensor();
            }
        }

        void UpdateSensor()
        {
        }
    }
}
