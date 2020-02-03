using UnityEngine;
using UnityEditor;
using Barracuda;
using MLAgents.Sensor;

namespace MLAgents
{
    /*
     This code is meant to modify the behavior of the inspector on Agent Components.
    */
    [CustomEditor(typeof(BehaviorParameters))]
    [CanEditMultipleObjects]
    public class BehaviorParametersEditor : Editor
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

            // Drawing the Behavior Parameters
            EditorGUI.BeginChangeCheck();
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(so.FindProperty("m_BehaviorName"));
            EditorGUILayout.PropertyField(so.FindProperty("m_BrainParameters"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_Model"), true);
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(so.FindProperty("m_InferenceDevice"), true);
            EditorGUI.indentLevel--;
            EditorGUILayout.PropertyField(so.FindProperty("m_BehaviorType"));
            EditorGUILayout.PropertyField(so.FindProperty("m_TeamID"));
            EditorGUILayout.PropertyField(so.FindProperty("m_useChildSensors"), true);
            // EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Heuristic"), true);
            EditorGUI.indentLevel--;
            if (EditorGUI.EndChangeCheck())
            {
                m_RequireReload = true;
            }
            DisplayFailedModelChecks();
            so.ApplyModifiedProperties();
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
            if(behaviorParameters.useChildSensors)
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
                var failedChecks = InferenceBrain.BarracudaModelParamLoader.CheckModel(
                    barracudaModel, brainParameters, sensorComponents);
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
