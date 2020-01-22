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
            var modelRunner = Academy.Instance.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_ModelRunner = modelRunner;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors, Action<AgentAction> action)
        {
            m_ModelRunner?.PutObservations(info, sensors, action);
        }

        /// <inheritdoc />
        public void DecideAction()
        {
            m_ModelRunner?.DecideBatch();
        }

        public void Dispose()
        {
        }
    }
}
