using UnityEngine;
using System.Collections.Generic;
using MLAgents.Sensor;
using System;

namespace MLAgents
{
    /// <summary>
    /// The Remote Policy only works when training.
    /// When training your Agents, the RemotePolicy will be controlled by Python.
    /// </summary>
    public class RemotePolicy : IPolicy
    {
        string m_BehaviorName;
        int m_AgentId;
        protected ICommunicator m_Communicator;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their Sensors.
        /// </summary>
        List<int[]> m_SensorShapes;

        /// <inheritdoc />
        public RemotePolicy(
            BrainParameters brainParameters,
            string behaviorName)
        {
            m_BehaviorName = behaviorName;
            m_Communicator = Academy.Instance.Communicator;
            m_Communicator.SubscribeBrain(m_BehaviorName, brainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            m_AgentId = info.episodeId;
            m_Communicator?.PutObservations(m_BehaviorName, info, sensors);
        }

        /// <inheritdoc />
        public float[] DecideAction()
        {
            m_Communicator?.DecideBatch();
            return m_Communicator?.GetActions(m_BehaviorName, m_AgentId);

        }

        public void Dispose()
        {
        }
    }
}
