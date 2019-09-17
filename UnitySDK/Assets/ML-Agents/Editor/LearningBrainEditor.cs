using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the LearningBrain class. Defines the default Inspector view for a
    /// LearningBrain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains. Also exposes a drag box for the Model that will be used by the
    /// LearningBrain.
    /// </summary>
    [CustomEditor(typeof(LearningBrain))]
    public class LearningBrainEditor : BrainEditor
    {
        private const string k_ModelPropName = "model";
        private const string k_InferenceDevicePropName = "inferenceDevice";
        private const float k_TimeBetweenModelReloads = 2f;
        // Time since the last reload of the model
        private float m_TimeSinceModelReload;
        // Whether or not the model needs to be reloaded
        private bool m_RequireReload;

        /// <summary>
        /// Called when the user opens the Inspector for the LearningBrain
        /// </summary>
        public void OnEnable()
        {
            m_RequireReload = true;
            EditorApplication.update += IncreaseTimeSinceLastModelReload;
        }

        /// <summary>
        /// Called when the user leaves the Inspector for the LearningBrain
        /// </summary>
        public void OnDisable()
        {
            EditorApplication.update -= IncreaseTimeSinceLastModelReload;
        }

        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Learning Brain", EditorStyles.boldLabel);
            var brain = (LearningBrain)target;
            var serializedBrain = serializedObject;
            EditorGUI.BeginChangeCheck();
            base.OnInspectorGUI();
            serializedBrain.Update();
            var tfGraphModel = serializedBrain.FindProperty(k_ModelPropName);
            EditorGUILayout.ObjectField(tfGraphModel);
            var inferenceDevice = serializedBrain.FindProperty(k_InferenceDevicePropName);
            EditorGUILayout.PropertyField(inferenceDevice);
            serializedBrain.ApplyModifiedProperties();
            if (EditorGUI.EndChangeCheck())
            {
                m_RequireReload = true;
            }
            if (m_RequireReload && m_TimeSinceModelReload > k_TimeBetweenModelReloads)
            {
                brain.ReloadModel();
                m_RequireReload = false;
                m_TimeSinceModelReload = 0;
            }
            // Display all failed checks
            var failedChecks = brain.GetModelFailedChecks();
            foreach (var check in failedChecks)
            {
                if (check != null)
                {
                    EditorGUILayout.HelpBox(check, MessageType.Warning);
                }
            }
        }

        /// <summary>
        /// Increases the time since last model reload by the deltaTime since the last Update call
        /// from the UnityEditor
        /// </summary>
        private void IncreaseTimeSinceLastModelReload()
        {
            m_TimeSinceModelReload += Time.deltaTime;
        }
    }
}
