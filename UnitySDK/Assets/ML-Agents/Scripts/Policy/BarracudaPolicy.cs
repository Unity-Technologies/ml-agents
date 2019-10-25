using UnityEngine;
using Barracuda;
using System.Collections.Generic;
using MLAgents.InferenceBrain;

namespace MLAgents
{
    public enum InferenceDevice
    {
        CPU = 0,
        GPU = 1
    }

    /// <summary>
    /// The Barracuda Policy uses a Barracuda Model to make decisions at
    /// every step. It uses a ModelRunner that is shared accross all
    /// Barracuda Policies that use the same model and inference devices.
    /// </summary>
    public class BarracudaPolicy : IPolicy
    {

        protected ModelRunner m_ModelRunner;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their Sensors.
        /// </summary>
        List<int[]> m_SensorShapes;

        /// <inheritdoc />
        public BarracudaPolicy(
            BrainParameters brainParameters,
            NNModel model,
            InferenceDevice inferenceDevice)
        {
            var aca = Object.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            var modelRunner = aca.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_ModelRunner = modelRunner;
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
#if DEBUG
            ValidateAgentSensorShapes(agent);
#endif
            m_ModelRunner?.PutObservations(agent);
        }

        /// <inheritdoc />
        public void DecideAction()
        {
            m_ModelRunner?.DecideBatch();
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
