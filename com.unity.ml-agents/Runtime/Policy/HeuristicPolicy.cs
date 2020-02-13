using System.Collections.Generic;
using System;

namespace MLAgents
{
    /// <summary>
    /// The Heuristic Policy uses a hards coded Heuristic method
    /// to take decisions each time the RequestDecision method is
    /// called.
    /// </summary>
    internal class HeuristicPolicy : IPolicy
    {
        Func<float[]> m_Heuristic;
        float[] m_LastDecision;

        /// <inheritdoc />
        public HeuristicPolicy(Func<float[]> heuristic)
        {
            m_Heuristic = heuristic;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            m_LastDecision = m_Heuristic.Invoke();
        }

        /// <inheritdoc />
        public float[] DecideAction()
        {
            return m_LastDecision;
        }

        public void Dispose()
        {
        }
    }
}
