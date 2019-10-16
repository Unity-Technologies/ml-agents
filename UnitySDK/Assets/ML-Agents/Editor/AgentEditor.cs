using UnityEngine;
using UnityEditor;
using Barracuda;

namespace MLAgents
{
    /*
     This code is meant to modify the behavior of the inspector on Agent Components.
    */
    [CustomEditor(typeof(Agent), true)]
    [CanEditMultipleObjects]
    public class AgentEditor : Editor
    {
        private const float k_TimeBetweenModelReloads = 2f;
        // Time since the last reload of the model
        private float m_TimeSinceModelReload;
        // Whether or not the model needs to be reloaded
        private bool m_RequireReload;

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
            var cameras = serializedAgent.FindProperty(
                "agentParameters.agentCameras");
            var renderTextures = serializedAgent.FindProperty(
                "agentParameters.agentRenderTextures");

            // Drawing the Behavior Parameters
            var brainParameters = serializedAgent.FindProperty("m_BrainParameters");
            brainParameters.isExpanded = EditorGUILayout.Foldout(brainParameters.isExpanded, "Behavior Parameters");
            if (brainParameters.isExpanded)
            {
                EditorGUI.BeginChangeCheck();
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(serializedAgent.FindProperty("m_BehaviorName"));
                EditorGUILayout.PropertyField(serializedAgent.FindProperty("m_BrainParameters"), true);
                EditorGUILayout.PropertyField(serializedAgent.FindProperty("m_Model"), true);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(serializedAgent.FindProperty("m_InferenceDevice"), true);
                EditorGUI.indentLevel--;
                EditorGUILayout.PropertyField(serializedAgent.FindProperty("m_UseHeuristic"));
                EditorGUI.indentLevel--;
                if (EditorGUI.EndChangeCheck())
                {
                    m_RequireReload = true;
                }
                DisplayFailedModelChecks();
            }
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);


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

        /// <summary>
        /// Must be called within OnEditorGUI()
        /// </summary>
        private void DisplayFailedModelChecks()
        {
            if (m_RequireReload && m_TimeSinceModelReload > k_TimeBetweenModelReloads)
            {
                m_RequireReload = false;
                m_TimeSinceModelReload = 0;
            }
            // Display all failed checks
            D.logEnabled = false;
            Model barracudaModel = null;
            var model = (NNModel)serializedObject.FindProperty("m_Model").objectReferenceValue;
            var brainParameters = ((Agent)target).brainParameters;
            if (model != null)
            {
                barracudaModel = ModelLoader.Load(model.Value);
            }
            var failedChecks = InferenceBrain.BarracudaModelParamLoader.CheckModel(
                barracudaModel, brainParameters);
            foreach (var check in failedChecks)
            {
                if (check != null)
                {
                    EditorGUILayout.HelpBox(check, MessageType.Warning);
                }
            }
        }
    }
}
