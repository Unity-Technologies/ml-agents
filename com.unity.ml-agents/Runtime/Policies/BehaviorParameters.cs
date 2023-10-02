using Unity.Sentis;
using System;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors.Reflection;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// Defines what type of behavior the Agent will be using
    /// </summary>
    [Serializable]
    public enum BehaviorType
    {
        /// <summary>
        /// The Agent will use the remote process for decision making.
        /// if unavailable, will use inference and if no model is provided, will use
        /// the heuristic.
        /// </summary>
        Default,

        /// <summary>
        /// The Agent will always use its heuristic
        /// </summary>
        HeuristicOnly,

        /// <summary>
        /// The Agent will always use inference with the provided
        /// neural network model.
        /// </summary>
        InferenceOnly
    }

    /// <summary>
    /// Options for controlling how the Agent class is searched for <see cref="ObservableAttribute"/>s.
    /// </summary>
    public enum ObservableAttributeOptions
    {
        /// <summary>
        /// All ObservableAttributes on the Agent will be ignored. This is the
        /// default behavior. If there are no  ObservableAttributes on the
        /// Agent, this will result in the fastest initialization time.
        /// </summary>
        Ignore,

        /// <summary>
        /// Only members on the declared class will be examined; members that are
        /// inherited are ignored. This is a reasonable tradeoff between
        /// performance and flexibility.
        /// </summary>
        /// <remarks>This corresponds to setting the
        /// [BindingFlags.DeclaredOnly](https://docs.microsoft.com/en-us/dotnet/api/system.reflection.bindingflags?view=netcore-3.1)
        /// when examining the fields and properties of the Agent class instance.
        /// </remarks>
        ExcludeInherited,

        /// <summary>
        /// All members on the class will be examined. This can lead to slower
        /// startup times.
        /// </summary>
        ExamineAll
    }

    /// <summary>
    /// A component for setting an <seealso cref="Agent"/> instance's behavior and
    /// brain properties.
    /// </summary>
    /// <remarks>At runtime, this component generates the agent's policy objects
    /// according to the settings you specified in the Editor.</remarks>
    [AddComponentMenu("ML Agents/Behavior Parameters", (int)MenuGroup.Default)]
    public class BehaviorParameters : MonoBehaviour
    {
        [HideInInspector, SerializeField]
        BrainParameters m_BrainParameters = new BrainParameters();

        /// <summary>
        /// Delegate for receiving events about Policy Updates.
        /// </summary>
        /// <param name="isInHeuristicMode">Whether or not the current policy is running in heuristic mode.</param>
        public delegate void PolicyUpdated(bool isInHeuristicMode);

        /// <summary>
        /// Event that fires when an Agent's policy is updated.
        /// </summary>
        internal event PolicyUpdated OnPolicyUpdated;

        /// <summary>
        /// The associated <see cref="Policies.BrainParameters"/> for this behavior.
        /// </summary>
        public BrainParameters BrainParameters
        {
            get { return m_BrainParameters; }
            internal set { m_BrainParameters = value; }
        }

        [HideInInspector, SerializeField]
        ModelAsset m_Model;

        /// <summary>
        /// The neural network model used when in inference mode.
        /// This should not be set at runtime; use <see cref="Agent.SetModel(string,Model,Policies.InferenceDevice)"/>
        /// to set it instead.
        /// </summary>
        public ModelAsset Model
        {
            get { return m_Model; }
            set { m_Model = value; UpdateAgentPolicy(); }
        }

        [HideInInspector, SerializeField]
        InferenceDevice m_InferenceDevice = InferenceDevice.Default;

        /// <summary>
        /// How inference is performed for this Agent's model.
        /// This should not be set at runtime; use <see cref="Agent.SetModel(string,Model,Policies.InferenceDevice)"/>
        /// to set it instead.
        /// </summary>
        public InferenceDevice InferenceDevice
        {
            get { return m_InferenceDevice; }
            set { m_InferenceDevice = value; UpdateAgentPolicy(); }
        }

        [HideInInspector, SerializeField]
        BehaviorType m_BehaviorType;

        /// <summary>
        /// The BehaviorType for the Agent.
        /// </summary>
        public BehaviorType BehaviorType
        {
            get { return m_BehaviorType; }
            set { m_BehaviorType = value; UpdateAgentPolicy(); }
        }

        [HideInInspector, SerializeField]
        string m_BehaviorName = "My Behavior";

        /// <summary>
        /// The name of this behavior, which is used as a base name. See
        /// <see cref="FullyQualifiedBehaviorName"/> for the full name.
        /// This should not be set at runtime; use <see cref="Agent.SetModel(string,Model,Policies.InferenceDevice)"/>
        /// to set it instead.
        /// </summary>
        public string BehaviorName
        {
            get { return m_BehaviorName; }
            set { m_BehaviorName = value; UpdateAgentPolicy(); }
        }

        /// <summary>
        /// The team ID for this behavior.
        /// </summary>
        [HideInInspector, SerializeField, FormerlySerializedAs("m_TeamID")]
        public int TeamId;
        // TODO properties here instead of Agent

        [FormerlySerializedAs("m_useChildSensors")]
        [HideInInspector]
        [SerializeField]
        [Tooltip("Use all Sensor components attached to child GameObjects of this Agent.")]
        bool m_UseChildSensors = true;

        [HideInInspector]
        [SerializeField]
        [Tooltip("Use all Actuator components attached to child GameObjects of this Agent.")]
        bool m_UseChildActuators = true;

        /// <summary>
        /// Whether or not to use all the sensor components attached to child GameObjects of the agent.
        /// Note that changing this after the Agent has been initialized will not have any effect.
        /// </summary>
        public bool UseChildSensors
        {
            get { return m_UseChildSensors; }
            set { m_UseChildSensors = value; }
        }

        [HideInInspector]
        [SerializeField]
        [Tooltip("Set action selection to deterministic, Only applies to inference from within unity.")]
        private bool m_DeterministicInference = false;

        /// <summary>
        /// Whether to select actions deterministically during inference from the provided neural network.
        /// </summary>
        public bool DeterministicInference
        {
            get { return m_DeterministicInference; }
            set { m_DeterministicInference = value; }
        }

        /// <summary>
        /// Whether or not to use all the actuator components attached to child GameObjects of the agent.
        /// Note that changing this after the Agent has been initialized will not have any effect.
        /// </summary>
        public bool UseChildActuators
        {
            get { return m_UseChildActuators; }
            set { m_UseChildActuators = value; }
        }

        [HideInInspector, SerializeField]
        ObservableAttributeOptions m_ObservableAttributeHandling = ObservableAttributeOptions.Ignore;

        /// <summary>
        /// Determines how the Agent class is searched for <see cref="ObservableAttribute"/>s.
        /// </summary>
        public ObservableAttributeOptions ObservableAttributeHandling
        {
            get { return m_ObservableAttributeHandling; }
            set { m_ObservableAttributeHandling = value; }
        }

        /// <summary>
        /// Returns the behavior name, concatenated with any other metadata (i.e. team id).
        /// </summary>
        public string FullyQualifiedBehaviorName
        {
            get { return m_BehaviorName + "?team=" + TeamId; }
        }

        void Awake()
        {
            OnPolicyUpdated += mode => { };
        }

        internal IPolicy GeneratePolicy(ActionSpec actionSpec, ActuatorManager actuatorManager)
        {
            switch (m_BehaviorType)
            {
                case BehaviorType.HeuristicOnly:
                    return new HeuristicPolicy(actuatorManager, actionSpec);
                case BehaviorType.InferenceOnly:
                    {
                        if (m_Model == null)
                        {
                            var behaviorType = BehaviorType.InferenceOnly.ToString();
                            throw new UnityAgentsException(
                                $"Can't use Behavior Type {behaviorType} without a model. " +
                                "Either assign a model, or change to a different Behavior Type."
                            );
                        }
                        return new SentisPolicy(actionSpec, actuatorManager, m_Model, m_InferenceDevice, m_BehaviorName, m_DeterministicInference);
                    }
                case BehaviorType.Default:
                    if (Academy.Instance.IsCommunicatorOn)
                    {
                        return new RemotePolicy(actionSpec, actuatorManager, FullyQualifiedBehaviorName);
                    }
                    if (m_Model != null)
                    {
                        return new SentisPolicy(actionSpec, actuatorManager, m_Model, m_InferenceDevice, m_BehaviorName, m_DeterministicInference);
                    }
                    else
                    {
                        return new HeuristicPolicy(actuatorManager, actionSpec);
                    }
                default:
                    return new HeuristicPolicy(actuatorManager, actionSpec);
            }
        }

        /// <summary>
        /// Query the behavior parameters in order to see if the Agent is running in Heuristic Mode.
        /// </summary>
        /// <returns>true if the Agent is running in Heuristic mode.</returns>
        public bool IsInHeuristicMode()
        {
            if (BehaviorType == BehaviorType.HeuristicOnly)
            {
                return true;
            }

            return BehaviorType == BehaviorType.Default &&
                ReferenceEquals(Model, null) &&
                (!Academy.IsInitialized ||
                    Academy.IsInitialized &&
                    !Academy.Instance.IsCommunicatorOn);
        }

        internal void UpdateAgentPolicy()
        {
            var agent = GetComponent<Agent>();
            if (agent == null)
            {
                return;
            }
            agent.ReloadPolicy();
            OnPolicyUpdated?.Invoke(IsInHeuristicMode());
        }
    }
}
