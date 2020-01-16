using MLAgents.Sensor;
using System.Collections.Generic;
using System;

namespace MLAgents
{

    /// <summary>
    /// The Heuristic Policy uses a hards coded Heuristic method
    /// to take decisions each time the RequestDecision method is
    /// called.
    /// </summary>
    public class HeuristicPolicy : IPolicy
    {
        Func<float[]> m_Heuristic;
        Action<AgentAction> m_ActionFunc;

        /// <inheritdoc />
        public HeuristicPolicy(Func<float[]> heuristic)
        {
            m_Heuristic = heuristic;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors, Action<AgentAction> action)
        {
            m_ActionFunc = action;
        }

        /// <inheritdoc />
        public void DecideAction()
        {
            if (m_ActionFunc != null)
            {
                m_ActionFunc.Invoke(new AgentAction { vectorActions = m_Heuristic.Invoke() });
                m_ActionFunc = null;
            }
        }

        public void Dispose()
        {

        }
    }
}
