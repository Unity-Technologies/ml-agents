using System;
using Unity.Barracuda;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// Where to perform inference.
    /// </summary>
    public enum InferenceDevice
    {
        /// <summary>
        /// CPU inference
        /// </summary>
        CPU = 0,

        /// <summary>
        /// GPU inference
        /// </summary>
        GPU = 1
    }

    /// <summary>
    /// The Barracuda Policy uses a Barracuda Model to make decisions at
    /// every step. It uses a ModelRunner that is shared across all
    /// Barracuda Policies that use the same model and inference devices.
    /// </summary>
    internal class BarracudaPolicy : IPolicy
    {
        protected ModelRunner m_ModelRunner;
        ActionBuffers m_LastActionBuffer;

        int m_AgentId;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their Sensors.
        /// </summary>
        List<int[]> m_SensorShapes;
        ActionSpec m_ActionSpec;

        /// <inheritdoc />
        public BarracudaPolicy(
            ActionSpec actionSpec,
            NNModel model,
            InferenceDevice inferenceDevice)
        {
            var modelRunner = Academy.Instance.GetOrCreateModelRunner(model, actionSpec, inferenceDevice);
            m_ModelRunner = modelRunner;
            m_ActionSpec = actionSpec;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            m_AgentId = info.episodeId;
            m_ModelRunner?.PutObservations(info, sensors);
        }

        /// <inheritdoc />
        public ref readonly ActionBuffers DecideAction()
        {
            if (m_ModelRunner == null)
            {
                m_LastActionBuffer = ActionBuffers.Empty;
            }
            else
            {
                m_ModelRunner?.DecideBatch();
                m_LastActionBuffer = m_ModelRunner.GetAction(m_AgentId);
            }
            return ref m_LastActionBuffer;
        }

        public void Dispose()
        {
        }
    }
}
