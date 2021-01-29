using Unity.Barracuda;
using System.Collections.Generic;
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

        int m_AgentId;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their Sensors.
        /// </summary>
        List<int[]> m_SensorShapes;

        private string m_BehaviorName;
        private BrainParameters m_BrainParameters;

        /// <summary>
        /// Whether or not we've tried to send analytics for this model. We only ever try to send once per policy,
        /// and do additional deduplication in the analytics code.
        /// </summary>
        private bool m_AnalyticsSent;

        /// <inheritdoc />
        public BarracudaPolicy(
            BrainParameters brainParameters,
            NNModel model,
            InferenceDevice inferenceDevice,
            string behaviorName
        )
        {
            var modelRunner = Academy.Instance.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_ModelRunner = modelRunner;
            m_BehaviorName = behaviorName;
            m_BrainParameters = brainParameters;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {

            if (!m_AnalyticsSent)
            {
                m_AnalyticsSent = true;
                Analytics.InferenceAnalytics.InferenceModelSet(
                    m_ModelRunner.Model,
                    m_BehaviorName,
                    m_ModelRunner.InferenceDevice,
                    sensors,
                    m_BrainParameters
                );
            }
            m_AgentId = info.episodeId;
            m_ModelRunner?.PutObservations(info, sensors);
        }

        /// <inheritdoc />
        public float[] DecideAction()
        {
            m_ModelRunner?.DecideBatch();
            return m_ModelRunner?.GetAction(m_AgentId);
        }

        public void Dispose()
        {
        }
    }
}
