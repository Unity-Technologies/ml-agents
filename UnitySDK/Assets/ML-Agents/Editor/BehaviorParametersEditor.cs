using UnityEngine;
using UnityEditor;
using Barracuda;

namespace MLAgents
{
    /*
     This code is meant to modify the behavior of the inspector on Agent Components.
    */
    [CustomEditor(typeof(BehaviorParameters))]
    [CanEditMultipleObjects]
    public class BehaviorParametersEditor : Editor
    {
        private const float k_TimeBetweenModelReloads = 2f;
        // Time since the last reload of the model
        private float m_TimeSinceModelReload;
        // Whether or not the model needs to be reloaded
        private bool m_RequireReload;

        public override void OnInspectorGUI()
        {
            var serializedObject = base.serializedObject;
            serializedObject.Update();

            // Drawing the Behavior Parameters
            EditorGUI.BeginChangeCheck();
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_BehaviorName"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_BrainParameters"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Model"), true);
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_InferenceDevice"), true);
            EditorGUI.indentLevel--;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_UseHeuristic"));
            // EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Heuristic"), true);
            EditorGUI.indentLevel--;
            if (EditorGUI.EndChangeCheck())
            {
                m_RequireReload = true;
            }
            DisplayFailedModelChecks();
            serializedObject.ApplyModifiedProperties();
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
            var brainParameters = ((BehaviorParameters)target).brainParameters;
            if (model != null)
            {
                barracudaModel = ModelLoader.Load(model.Value);
            }
            if (brainParameters != null)
            {
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
}
