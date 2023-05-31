#if MLA_INPUT_SYSTEM
using Unity.MLAgents.Extensions.Input;
using UnityEditor;

namespace Unity.MLAgents.Extensions.Editor.Input
{
    [CustomEditor(typeof(InputActuatorComponent))]
    internal class InputActuatorComponentEditor : UnityEditor.Editor
    {
        const string k_ActionSpecName = "m_ActionSpec";

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();
            InputActuatorComponent o = so.targetObject as InputActuatorComponent;
            _ = o.ActionSpec;
            EditorGUI.indentLevel++;
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(so.FindProperty(k_ActionSpecName));
            EditorGUI.EndDisabledGroup();
            EditorGUI.indentLevel--;
        }
    }
}
#endif // MLA_INPUT_SYSTEM
