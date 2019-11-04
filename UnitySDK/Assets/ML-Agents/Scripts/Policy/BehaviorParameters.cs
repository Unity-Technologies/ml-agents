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

        [HideInInspector]
        [SerializeField]
        BrainParameters m_BrainParameters = new BrainParameters();
        [HideInInspector] [SerializeField]
        NNModel m_Model;
        [HideInInspector] [SerializeField]
        InferenceDevice m_InferenceDevice;
        [HideInInspector] [SerializeField]
        bool m_UseHeuristic;
        [HideInInspector] [SerializeField]
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
            if (m_UseHeuristic)
            {
                return new HeuristicPolicy(heuristic);
            }
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
