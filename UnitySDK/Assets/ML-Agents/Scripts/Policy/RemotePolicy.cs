using UnityEngine;
using System.Collections.Generic;

namespace MLAgents
{
    /// <summary>
    /// The Remote Policy only works when training.
    /// When training your Agents, the RemotePolicy will be controlled by Python.
    /// </summary>
    public class RemotePolicy : IPolicy
    {
        string m_BehaviorName;
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
            var aca = Object.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            m_Communicator = aca.Communicator;
            aca.Communicator.SubscribeBrain(m_BehaviorName, brainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
#if DEBUG
            ValidateAgentSensorShapes(agent);
#endif
            m_Communicator?.PutObservations(m_BehaviorName, agent);
        }

        /// <inheritdoc />
        public void DecideAction()
        {
            m_Communicator?.DecideBatch();
        }

        /// <summary>
        /// Check that the Agent Sensors are the same shape as the the other Agents using the same Brain.
        /// If this is the first Agent being checked, its Sensor sizes will be saved.
        /// </summary>
        /// <param name="agent">The Agent to check</param>
        void ValidateAgentSensorShapes(Agent agent)
        {
            if (m_SensorShapes == null)
            {
                m_SensorShapes = new List<int[]>(agent.sensors.Count);
                // First agent, save the sensor sizes
                foreach (var sensor in agent.sensors)
                {
                    m_SensorShapes.Add(sensor.GetFloatObservationShape());
                }
            }
            else
            {
                // Check for compatibility with the other Agents' Sensors
                // TODO make sure this only checks once per agent
                Debug.Assert(m_SensorShapes.Count == agent.sensors.Count, $"Number of Sensors must match. {m_SensorShapes.Count} != {agent.sensors.Count}");
                for (var i = 0; i < m_SensorShapes.Count; i++)
                {
                    var cachedShape = m_SensorShapes[i];
                    var sensorShape = agent.sensors[i].GetFloatObservationShape();
                    Debug.Assert(cachedShape.Length == sensorShape.Length, "Sensor dimensions must match.");
                    for (var j = 0; j < cachedShape.Length; j++)
                    {
                        Debug.Assert(cachedShape[j] == sensorShape[j], "Sensor sizes much match.");
                    }
                }
            }
        }

        public void Dispose()
        {
        }
    }
}
