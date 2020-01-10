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
            var aca = GameObject.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            m_Communicator = aca.Communicator;
            aca.Communicator.SubscribeBrain(m_BehaviorName, brainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors, Action<AgentAction> action)
        {
#if DEBUG
            ValidateAgentSensorShapes(info);
#endif
            m_Communicator?.PutObservations(m_BehaviorName, info, sensors, action);
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
        void ValidateAgentSensorShapes(AgentInfo info)
        {
            if (m_SensorShapes == null)
            {
                m_SensorShapes = new List<int[]>(info.observations.Count);
                // First agent, save the sensor sizes
                foreach (var obs in info.observations)
                {
                    m_SensorShapes.Add(obs.Shape);
                }
            }
            else
            {
                // Check for compatibility with the other Agents' Sensors
                // TODO make sure this only checks once per agent
                Debug.Assert(m_SensorShapes.Count == info.observations.Count, $"Number of Sensors must match. {m_SensorShapes.Count} != {info.observations.Count}");
                for (var i = 0; i < m_SensorShapes.Count; i++)
                {
                    var cachedShape = m_SensorShapes[i];
                    var sensorShape = info.observations[i].Shape;
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
