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

        private string m_BehaviorName;
        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their sensors.
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
            m_BatchedDecisionMaker = aca.Communicator;
            aca.Communicator.SubscribeBrain(m_BehaviorName, brainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
#if DEBUG
            ValidateAgentSensorShapes(agent);
#endif
            m_BatchedDecisionMaker?.PutObservations(m_BehaviorName, agent);
        }

        /// <inheritdoc />
        public void DecideAction()
        {
            m_BatchedDecisionMaker?.DecideBatch();
        }

        /// <summary>
        /// Check that the Agent sensors are the same shape as the the other Agents using the same Brain.
        /// If this is the first Agent being checked, its Sensor sizes will be saved.
        /// </summary>
        /// <param name="agent">The Agent to check</param>
        private void ValidateAgentSensorShapes(Agent agent)
        {
            if (m_SensorShapes == null)
            {
                m_SensorShapes = new List<int[]>(agent.m_Sensors.Count);
                // First agent, save the sensor sizes
                foreach (var sensor in agent.m_Sensors)
                {
                    m_SensorShapes.Add(sensor.GetFloatObservationShape());
                }
            }
            else
            {
                // Check for compatibility with the other Agents' sensors
                // TODO make sure this only checks once per agent
                Debug.Assert(m_SensorShapes.Count == agent.m_Sensors.Count, $"Number of sensors must match. {m_SensorShapes.Count} != {agent.m_Sensors.Count}");
                for (var i = 0; i < m_SensorShapes.Count; i++)
                {
                    var cachedShape = m_SensorShapes[i];
                    var sensorShape = agent.m_Sensors[i].GetFloatObservationShape();
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
