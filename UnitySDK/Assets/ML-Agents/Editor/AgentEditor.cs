using UnityEngine;
using UnityEditor;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
    [CustomEditor(typeof(Agent), true)]
    [CanEditMultipleObjects]
    public class AgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            var serializedAgent = serializedObject;
            serializedAgent.Update();

            var brain = serializedAgent.FindProperty("brain");
            var actionsPerDecision = serializedAgent.FindProperty(
                "agentParameters.numberOfActionsBetweenDecisions");
            var maxSteps = serializedAgent.FindProperty(
                "agentParameters.maxStep");
            var isResetOnDone = serializedAgent.FindProperty(
                "agentParameters.resetOnDone");
            var isOdd = serializedAgent.FindProperty(
                "agentParameters.onDemandDecision");
            var cameras = serializedAgent.FindProperty(
                "agentParameters.agentCameras");
            var renderTextures = serializedAgent.FindProperty(
                "agentParameters.agentRenderTextures");

            EditorGUILayout.PropertyField(brain);

            if (cameras.arraySize > 0 && renderTextures.arraySize > 0)
            {
                EditorGUILayout.HelpBox("Brain visual observations created by first getting all cameras then all render textures.", MessageType.Info);
            }

            EditorGUILayout.LabelField("Agent Cameras");
            for (var i = 0; i < cameras.arraySize; i++)
            {
                EditorGUILayout.PropertyField(
                    cameras.GetArrayElementAtIndex(i),
                    new GUIContent("Camera " + (i + 1) + ": "));
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Add Camera", EditorStyles.miniButton))
            {
                cameras.arraySize++;
            }

            if (GUILayout.Button("Remove Camera", EditorStyles.miniButton))
            {
                cameras.arraySize--;
            }

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField("Agent RenderTextures");
            for (var i = 0; i < renderTextures.arraySize; i++)
            {
                EditorGUILayout.PropertyField(
                    renderTextures.GetArrayElementAtIndex(i),
                    new GUIContent("RenderTexture " + (i + 1) + ": "));
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Add RenderTextures", EditorStyles.miniButton))
            {
                renderTextures.arraySize++;
            }

            if (GUILayout.Button("Remove RenderTextures", EditorStyles.miniButton))
            {
                renderTextures.arraySize--;
            }

            EditorGUILayout.EndHorizontal();


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
