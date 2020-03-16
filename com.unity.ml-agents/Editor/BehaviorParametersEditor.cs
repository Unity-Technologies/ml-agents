using MLAgents.Sensors;
using UnityEditor;
using Barracuda;
using MLAgents.Policies;
using UnityEngine;

namespace MLAgents.Editor
{
    /*
     This code is meant to modify the behavior of the inspector on Agent Components.
    */
    [CustomEditor(typeof(BehaviorParameters))]
    [CanEditMultipleObjects]
    internal class BehaviorParametersEditor : UnityEditor.Editor
    {
        const float k_TimeBetweenModelReloads = 2f;
        // Time since the last reload of the model
        float m_TimeSinceModelReload;
        // Whether or not the model needs to be reloaded
        bool m_RequireReload;

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();
            bool needPolicyUpdate; // Whether the name, model, inference device, or BehaviorType changed.

            // Drawing the Behavior Parameters
            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck(); // global

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_BehaviorName"));
            }
            needPolicyUpdate = EditorGUI.EndChangeCheck();

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_BrainParameters"), true);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_Model"), true);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(so.FindProperty("m_InferenceDevice"), true);
                EditorGUI.indentLevel--;
            }
            needPolicyUpdate = needPolicyUpdate || EditorGUI.EndChangeCheck();

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_BehaviorType"));
            }
            needPolicyUpdate = needPolicyUpdate || EditorGUI.EndChangeCheck();

            EditorGUILayout.PropertyField(so.FindProperty("TeamId"));
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_UseChildSensors"), true);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.indentLevel--;
            m_RequireReload = EditorGUI.EndChangeCheck();
            DisplayFailedModelChecks();
            so.ApplyModifiedProperties();

            if (needPolicyUpdate)
            {
                UpdateAgentPolicy();
            }
        }

        /// <summary>
        /// Must be called within OnEditorGUI()
        /// </summary>
        void DisplayFailedModelChecks()
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
            var behaviorParameters = (BehaviorParameters)target;
            SensorComponent[] sensorComponents;
            if (behaviorParameters.useChildSensors)
            {
                sensorComponents = behaviorParameters.GetComponentsInChildren<SensorComponent>();
            }
            else
            {
                sensorComponents = behaviorParameters.GetComponents<SensorComponent>();
            }
            var brainParameters = behaviorParameters.brainParameters;
            if (model != null)
            {
                barracudaModel = ModelLoader.Load(model);
            }
            if (brainParameters != null)
            {
                var failedChecks = Inference.BarracudaModelParamLoader.CheckModel(
                    barracudaModel, brainParameters, sensorComponents, behaviorParameters.behaviorType
                );
                foreach (var check in failedChecks)
                {
                    if (check != null)
                    {
                        EditorGUILayout.HelpBox(check, MessageType.Warning);
                    }
                }
            }
        }

        void UpdateAgentPolicy()
        {
            var behaviorParameters = (BehaviorParameters)target;
            behaviorParameters.UpdateAgentPolicy();
        }
    }
}
