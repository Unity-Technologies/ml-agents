using System.Collections.Generic;
using System.Diagnostics;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Analytics;


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
        bool m_AnalyticsSent;

        internal ICommunicator m_Communicator;

        /// <summary>
        /// List of actuators, only used for analytics
        /// </summary>
        private IList<IActuator> m_Actuators;

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
            SendAnalytics(sensors);
            m_AgentId = info.episodeId;
            m_Communicator?.PutObservations(m_FullyQualifiedBehaviorName, info, sensors);
        }

        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        void SendAnalytics(IList<ISensor> sensors)
        {
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
