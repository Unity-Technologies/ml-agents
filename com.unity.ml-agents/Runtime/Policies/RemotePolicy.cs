using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
#if MLA_UNITY_ANALYTICS
using Unity.MLAgents.Analytics;
#endif


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
#if MLA_UNITY_ANALYTICS
        private bool m_AnalyticsSent = false;
#endif

        internal ICommunicator m_Communicator;

        /// <summary>
        /// List of actuators, only used for analytics
        /// </summary>
        private IList<IActuator> m_Actuators;

        /// <inheritdoc />
        public RemotePolicy(
            ActionSpec actionSpec,
            IList<IActuator> actuators,
            string fullyQualifiedBehaviorName)
        {
            m_FullyQualifiedBehaviorName = fullyQualifiedBehaviorName;
            m_Communicator = Academy.Instance.Communicator;
            m_Communicator?.SubscribeBrain(m_FullyQualifiedBehaviorName, actionSpec);
            m_ActionSpec = actionSpec;
            m_Actuators = actuators;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
#if MLA_UNITY_ANALYTICS
            if (!m_AnalyticsSent)
            {
                m_AnalyticsSent = true;
                TrainingAnalytics.RemotePolicyInitialized(
                    m_FullyQualifiedBehaviorName,
                    sensors,
                    m_ActionSpec,
                    m_Actuators
                );
            }
#endif
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
