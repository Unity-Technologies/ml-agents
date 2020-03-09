using Barracuda;
using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents.Policies
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
    /// The Factory to generate policies.
    /// </summary>
    [AddComponentMenu("ML Agents/Behavior Parameters", (int)MenuGroup.Default)]
    public class BehaviorParameters : MonoBehaviour
    {
        [HideInInspector, SerializeField]
        BrainParameters m_BrainParameters = new BrainParameters();

        /// <summary>
        /// The associated <see cref="BrainParameters"/> for this behavior.
        /// </summary>
        public BrainParameters brainParameters
        {
            get { return m_BrainParameters; }
            internal set { m_BrainParameters = value; }
        }

        [HideInInspector, SerializeField]
        NNModel m_Model;

        /// <summary>
        /// The neural network model used when in inference mode.
        /// This cannot be set directly; use <see cref="Agent.GiveModel(string,NNModel,InferenceDevice)"/>
        /// to set it.
        /// </summary>
        public NNModel model
        {
            get { return m_Model; }
        }

        [HideInInspector, SerializeField]
        InferenceDevice m_InferenceDevice;

        /// <summary>
        /// How inference is performed for this Agent's model.
        /// This cannot be set directly; use <see cref="Agent.GiveModel(string,NNModel,InferenceDevice)"/>
        /// to set it.
        /// </summary>
        public InferenceDevice inferenceDevice
        {
            get { return m_InferenceDevice; }
        }

        [HideInInspector, SerializeField]
        BehaviorType m_BehaviorType;

        /// <summary>
        /// The BehaviorType for the Agent.
        /// This cannot be set directly; use <see cref="Agent.SetBehaviorType(BehaviorType)"/>
        /// to set it.
        /// </summary>
        public BehaviorType behaviorType
        {
            get { return m_BehaviorType; }
            internal set { m_BehaviorType = value; }
        }

        [HideInInspector, SerializeField]
        string m_BehaviorName = "My Behavior";

        /// <summary>
        /// The name of this behavior, which is used as a base name. See
        /// <see cref="fullyQualifiedBehaviorName"/> for the full name.
        /// This cannot be set directly; use <see cref="Agent.GiveModel(string,NNModel,InferenceDevice)"/>
        /// to set it.
        /// </summary>
        public string behaviorName
        {
            get { return m_BehaviorName; }
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

        /// <summary>
        /// Whether or not to use all the sensor components attached to child GameObjects of the agent.
        /// </summary>
        public bool useChildSensors
        {
            get { return m_UseChildSensors; }
            internal set { m_UseChildSensors = value; } // TODO make public, don't allow changes at runtime
        }

        /// <summary>
        /// Returns the behavior name, concatenated with any other metadata (i.e. team id).
        /// </summary>
        public string fullyQualifiedBehaviorName
        {
            get { return m_BehaviorName + "?team=" + TeamId; }
        }

        internal IPolicy GeneratePolicy(Func<float[]> heuristic)
        {
            switch (m_BehaviorType)
            {
                case BehaviorType.HeuristicOnly:
                    return new HeuristicPolicy(heuristic);
                case BehaviorType.InferenceOnly:
                    return new BarracudaPolicy(m_BrainParameters, m_Model, m_InferenceDevice);
                case BehaviorType.Default:
                    if (Academy.Instance.IsCommunicatorOn)
                    {
                        return new RemotePolicy(m_BrainParameters, fullyQualifiedBehaviorName);
                    }
                    if (m_Model != null)
                    {
                        return new BarracudaPolicy(m_BrainParameters, m_Model, m_InferenceDevice);
                    }
                    else
                    {
                        return new HeuristicPolicy(heuristic);
                    }
                default:
                    return new HeuristicPolicy(heuristic);
            }
        }

        /// <summary>
        /// Updates the model and related details for this behavior.
        /// </summary>
        /// <param name="newBehaviorName">New name for the behavior.</param>
        /// <param name="model">New neural network model for this behavior.</param>
        /// <param name="inferenceDevice">New inference device for this behavior.</param>
        internal void SetModel(
            string newBehaviorName,
            NNModel model,
            InferenceDevice inferenceDevice = InferenceDevice.CPU)
        {
            m_Model = model;
            m_InferenceDevice = inferenceDevice;
            m_BehaviorName = newBehaviorName;
        }
    }
}
