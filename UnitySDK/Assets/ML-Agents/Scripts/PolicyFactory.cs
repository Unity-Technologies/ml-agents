using Barracuda;
using System;
using UnityEngine;

namespace MLAgents
{

    public delegate float[] Heuristic();
    /// <summary>
    /// The Factory to generate policies. 
    /// </summary>
    public class PolicyFactory : MonoBehaviour
    {

        [HideInInspector]
        [SerializeField]
        private BrainParameters m_BrainParameters = new BrainParameters();
        [HideInInspector] [SerializeField] private NNModel m_Model;
        [HideInInspector] [SerializeField] private InferenceDevice m_InferenceDevice;
        [HideInInspector] [SerializeField] private bool m_UseHeuristic;
        [HideInInspector] [SerializeField] private string m_BehaviorName = "My Behavior";

        [HideInInspector] [SerializeField] private Heuristic m_Heuristic;

        [HideInInspector]
        public BrainParameters brainParameters
        {
            get { return m_BrainParameters; }
        }

        [HideInInspector]
        public string behaviorName
        {
            get { return m_BehaviorName; }
        }

        public IPolicy GeneratePolicy()
        {
            if (m_Model == null || m_UseHeuristic)
            {
                return new HeuristicPolicy(m_Heuristic);
            }
            if (FindObjectOfType<Academy>().IsCommunicatorOn)
            {
                return new RemotePolicy(m_BrainParameters, m_BehaviorName);
            }
            else
            {
                return new BarracudaPolicy(m_BrainParameters, m_Model, m_InferenceDevice);
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
