using UnityEngine;
using Barracuda;
using System.Collections.Generic;
using MLAgents.InferenceBrain;
using System;
using MLAgents.Sensor;

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
            var aca = GameObject.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            var modelRunner = aca.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_ModelRunner = modelRunner;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors, Action<AgentAction> action)
        {
#if DEBUG
            ValidateAgentSensorShapes(info);
#endif
            m_ModelRunner?.PutObservations(info, sensors, action);
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
