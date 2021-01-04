using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// The Remote Policy only works when training.
    /// When training your Agents, the RemotePolicy will be controlled by Python.
    /// </summary>
    internal class RemotePolicy : IPolicy
    {
        int m_AgentId;
        string m_FullyQualifiedBehaviorName;
        ActionSpec m_ActionSpec;
        ActionBuffers m_LastActionBuffer;

        internal ICommunicator m_Communicator;

        /// <inheritdoc />
        public RemotePolicy(
            ActionSpec actionSpec,
            string fullyQualifiedBehaviorName)
        {
            m_FullyQualifiedBehaviorName = fullyQualifiedBehaviorName;
            m_Communicator = Academy.Instance.Communicator;
            m_Communicator.SubscribeBrain(m_FullyQualifiedBehaviorName, actionSpec);
            m_ActionSpec = actionSpec;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            m_AgentId = info.episodeId;
            m_Communicator?.PutObservations(m_FullyQualifiedBehaviorName, info, sensors);
        }

        /// <inheritdoc />
        public ref readonly ActionBuffers DecideAction()
        {
            m_Communicator?.DecideBatch();
            var actions = m_Communicator?.GetActions(m_FullyQualifiedBehaviorName, m_AgentId);
            m_LastActionBuffer = actions == null ? ActionBuffers.Empty : (ActionBuffers)actions;
            return ref m_LastActionBuffer;
        }

        public void Dispose()
        {
        }
    }
}
