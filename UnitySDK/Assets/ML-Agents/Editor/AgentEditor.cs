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

            var actionsPerDecision = serializedAgent.FindProperty(
                "agentParameters.numberOfActionsBetweenDecisions");
            var maxSteps = serializedAgent.FindProperty(
                "agentParameters.maxStep");
            var isResetOnDone = serializedAgent.FindProperty(
                "agentParameters.resetOnDone");
            var isOdd = serializedAgent.FindProperty(
                "agentParameters.onDemandDecision");



            EditorGUILayout.PropertyField(
                maxSteps,
                new GUIContent(
                    "Max Step", "The per-agent maximum number of steps."));
            EditorGUILayout.PropertyField(
                isResetOnDone,
                new GUIContent(
                    "Reset On Done",
                    "If checked, the agent will reset on done. Else, AgentOnDone() will be called."));
            EditorGUILayout.PropertyField(
                isOdd,
                new GUIContent(
                    "On Demand Decisions",
                    "If checked, you must manually request decisions."));
            if (!isOdd.boolValue)
            {
                EditorGUILayout.PropertyField(
                    actionsPerDecision,
                    new GUIContent(
                        "Decision Interval",
                        "The agent will automatically request a decision every X" +
                        " steps and perform an action at every step."));
                actionsPerDecision.intValue = Mathf.Max(1, actionsPerDecision.intValue);
            }

            serializedAgent.ApplyModifiedProperties();

            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            base.OnInspectorGUI();
        }
    }
}
