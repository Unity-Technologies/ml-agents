using UnityEditor;
using Unity.MLAgents.Integrations.Match3;
namespace Unity.MLAgents.Editor
{
    [CustomEditor(typeof(Match3ActuatorComponent), editorForChildClasses: true)]
    [CanEditMultipleObjects]
    internal class Match3ActuatorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            var component = (Match3ActuatorComponent)target;
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
                EditorGUILayout.PropertyField(so.FindProperty("m_ActuatorName"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_RandomSeed"), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty("m_ForceHeuristic"), true);

            var requireSensorUpdate = EditorGUI.EndChangeCheck();
            so.ApplyModifiedProperties();

            if (requireSensorUpdate)
            {
                UpdateActuator();
            }
        }

        void UpdateActuator()
        {
        }
    }
}
