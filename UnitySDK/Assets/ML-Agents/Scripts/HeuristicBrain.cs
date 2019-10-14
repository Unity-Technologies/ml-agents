using UnityEngine;
using Barracuda;
using MLAgents.InferenceBrain;
using System;

namespace MLAgents
{

    public class HeuristicBrain : IBrain
    {
        private Func<float[]> m_Heuristic;
        private Agent m_Agent;

        /// <inheritdoc />
        public HeuristicBrain(Func<float[]> heuristic)
        {
            m_Heuristic = heuristic;
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
            m_Agent = agent;
        }

        public void Dispose()
        {

        }

        public void DecideAction()
        {
            if (m_Agent != null)
            {
                m_Agent.UpdateVectorAction(m_Heuristic.Invoke());
            }
        }
    }
}
