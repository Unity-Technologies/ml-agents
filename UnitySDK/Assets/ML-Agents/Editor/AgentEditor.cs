using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /*
     This code is meant to modify the behavior of the inspector on Agent Components.
    */
    [CustomEditor(typeof(Agent), true)]
    [CanEditMultipleObjects]
    public class AgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            var serializedAgent = serializedObject;
            serializedAgent.Update();

            var maxSteps = serializedAgent.FindProperty(
                "agentParameters.maxStep");

            EditorGUILayout.PropertyField(
                maxSteps,
                new GUIContent(
                    "Max Step", "The per-agent maximum number of steps."));

            serializedAgent.ApplyModifiedProperties();

            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            base.OnInspectorGUI();
        }
    }
}
