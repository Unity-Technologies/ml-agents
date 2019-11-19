using Barracuda;
using System;
using UnityEngine;

namespace MLAgents
{

    /// <summary>
    /// The Factory to generate policies. 
    /// </summary>
    public class BehaviorParameters : MonoBehaviour
    {

        [Serializable]
        private enum BehaviorType
        {
            DEFAULT,
            HEURISTIC_ONLY,
            INFERENCE_ONLY
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

        public BrainParameters brainParameters
        {
            get { return m_BrainParameters; }
        }

        public string behaviorName
        {
            get { return m_BehaviorName; }
        }

        public IPolicy GeneratePolicy(Func<float[]> heuristic)
        {
            switch (m_BehaviorType)
            {
                case BehaviorType.HEURISTIC_ONLY:
                    return new HeuristicPolicy(heuristic);
                case BehaviorType.INFERENCE_ONLY:
                    return new BarracudaPolicy(m_BrainParameters, m_Model, m_InferenceDevice);
                case BehaviorType.DEFAULT:
                    if (FindObjectOfType<Academy>().IsCommunicatorOn)
                    {
                        return new RemotePolicy(m_BrainParameters, m_BehaviorName);
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

        public void GiveModel(
            string behaviorName,
            NNModel model,
            InferenceDevice inferenceDevice = InferenceDevice.CPU)
        {
            m_Model = model;
            m_InferenceDevice = inferenceDevice;
            m_BehaviorName = behaviorName;
        }
    }
}
