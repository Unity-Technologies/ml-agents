using System.Collections.Generic;
using UnityEditor;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using CheckTypeEnum = Unity.MLAgents.Inference.BarracudaModelParamLoader.FailedCheck.CheckTypeEnum;

namespace Unity.MLAgents.Editor
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
        const string k_BehaviorName = "m_BehaviorName";
        const string k_BrainParametersName = "m_BrainParameters";
        const string k_ModelName = "m_Model";
        const string k_InferenceDeviceName = "m_InferenceDevice";
        const string k_BehaviorTypeName = "m_BehaviorType";
        const string k_TeamIdName = "TeamId";
        const string k_UseChildSensorsName = "m_UseChildSensors";
        const string k_ObservableAttributeHandlingName = "m_ObservableAttributeHandling";

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();
            bool needPolicyUpdate; // Whether the name, model, inference device, or BehaviorType changed.

            var behaviorParameters = (BehaviorParameters)target;
            var agent = behaviorParameters.gameObject.GetComponent<Agent>();
            if (agent == null)
            {
                EditorGUILayout.HelpBox(
                    "No Agent is associated with this Behavior Parameters. Attach an Agent to " +
                    "this GameObject to configure your Agent with these behavior parameters.",
                    MessageType.Warning);
            }

            // Drawing the Behavior Parameters
            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck(); // global

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty(k_BehaviorName));
            }
            needPolicyUpdate = EditorGUI.EndChangeCheck();

            EditorGUI.BeginChangeCheck();
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty(k_BrainParametersName), true);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty(k_ModelName), true);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(so.FindProperty(k_InferenceDeviceName), true);
                EditorGUI.indentLevel--;
            }
            needPolicyUpdate = needPolicyUpdate || EditorGUI.EndChangeCheck();

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.PropertyField(so.FindProperty(k_BehaviorTypeName));
            }
            needPolicyUpdate = needPolicyUpdate || EditorGUI.EndChangeCheck();

            EditorGUILayout.PropertyField(so.FindProperty(k_TeamIdName));
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty(k_UseChildSensorsName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_ObservableAttributeHandlingName), true);
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
            var model = (NNModel)serializedObject.FindProperty(k_ModelName).objectReferenceValue;
            var behaviorParameters = (BehaviorParameters)target;

            // Grab the sensor components, since we need them to determine the observation sizes.
            // TODO make these methods of BehaviorParameters
            var agent = behaviorParameters.gameObject.GetComponent<Agent>();
            if (agent == null)
            {
                return;
            }
            agent.sensors = new List<ISensor>();
            agent.InitializeSensors();
            var sensors = agent.sensors.ToArray();

            ActuatorComponent[] actuatorComponents;
            if (behaviorParameters.UseChildActuators)
            {
                actuatorComponents = behaviorParameters.GetComponentsInChildren<ActuatorComponent>();
            }
            else
            {
                actuatorComponents = behaviorParameters.GetComponents<ActuatorComponent>();
            }

            // Get the total size of the sensors generated by ObservableAttributes.
            // If there are any errors (e.g. unsupported type, write-only properties), display them too.
            int observableAttributeSensorTotalSize = 0;
            if (agent != null && behaviorParameters.ObservableAttributeHandling != ObservableAttributeOptions.Ignore)
            {
                List<string> observableErrors = new List<string>();
                observableAttributeSensorTotalSize = ObservableAttribute.GetTotalObservationSize(agent, false, observableErrors);
                foreach (var check in observableErrors)
                {
                    EditorGUILayout.HelpBox(check, MessageType.Warning);
                }
            }

            var brainParameters = behaviorParameters.BrainParameters;
            if (model != null)
            {
                barracudaModel = ModelLoader.Load(model);
            }
            if (brainParameters != null)
            {
                var failedChecks = Inference.BarracudaModelParamLoader.CheckModel(
                    barracudaModel, brainParameters, sensors, actuatorComponents,
                    observableAttributeSensorTotalSize, behaviorParameters.BehaviorType
                );
                foreach (var check in failedChecks)
                {
                    if (check != null)
                    {
                        switch (check.CheckType)
                        {
                            case CheckTypeEnum.Info:
                                EditorGUILayout.HelpBox(check.Message, MessageType.Info);
                                break;
                            case CheckTypeEnum.Warning:
                                EditorGUILayout.HelpBox(check.Message, MessageType.Warning);
                                break;
                            case CheckTypeEnum.Error:
                                EditorGUILayout.HelpBox(check.Message, MessageType.Error);
                                break;
                            default:
                                break;
                        }
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
