using Barracuda;
using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// The Factory to generate policies.
    /// </summary>
    [AddComponentMenu("ML Agents/Behavior Parameters", (int)MenuGroup.Default)]
    public class BehaviorParameters : MonoBehaviour
    {
        [Serializable]
        enum BehaviorType
        {
            Default,
            HeuristicOnly,
            InferenceOnly
        }

        [HideInInspector]
        [SerializeField]
        BrainParameters m_BrainParameters = new BrainParameters();
        [HideInInspector]
        [SerializeField]
        NNModel m_Model;
        [HideInInspector]
        [SerializeField]
        InferenceDevice m_InferenceDevice;
        [HideInInspector]
        [SerializeField]
        BehaviorType m_BehaviorType;
        [HideInInspector]
        [SerializeField]
        string m_BehaviorName = "My Behavior";

        /// <summary>
        /// The team ID for this behavior.
        /// </summary>
        [HideInInspector]
        [SerializeField]
        public int m_TeamID;
        [FormerlySerializedAs("m_useChildSensors")]
        [HideInInspector]
        [SerializeField]
        [Tooltip("Use all Sensor components attached to child GameObjects of this Agent.")]
        bool m_UseChildSensors = true;

        /// <summary>
        /// The associated <see cref="BrainParameters"/> for this behavior.
        /// </summary>
        public BrainParameters brainParameters
        {
            get { return m_BrainParameters; }
        }

        /// <summary>
        /// Whether or not to use all the sensor components attached to child GameObjects of the agent.
        /// </summary>
        public bool useChildSensors
        {
            get { return m_UseChildSensors; }
        }

        /// <summary>
        /// The name of this behavior, which is used as a base name. See
        /// <see cref="fullyQualifiedBehaviorName"/> for the full name.
        /// </summary>
        public string behaviorName
        {
            get { return m_BehaviorName; }
        }

        /// <summary>
        /// Returns the behavior name, concatenated with any other metadata (i.e. team id).
        /// </summary>
        public string fullyQualifiedBehaviorName
        {
            get { return m_BehaviorName + "?team=" + m_TeamID; }
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
        public void GiveModel(
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
