using Unity.Barracuda;
using System.Collections.Generic;
using System.Diagnostics;
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
        /// Default inference. This is currently the same as Burst, but may change in the future.
        /// </summary>
        Default = 0,

        /// <summary>
        /// GPU inference. Corresponds to WorkerFactory.Type.ComputePrecompiled in Barracuda.
        /// </summary>
        GPU = 1,

        /// <summary>
        /// CPU inference using Burst. Corresponds to WorkerFactory.Type.CSharpBurst in Barracuda.
        /// </summary>
        Burst = 2,

        /// <summary>
        /// CPU inference. Corresponds to in WorkerFactory.Type.CSharp Barracuda.
        /// Burst is recommended instead; this is kept for legacy compatibility.
        /// </summary>
        CPU = 3,
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

        private string m_BehaviorName;

        /// <summary>
        /// List of actuators, only used for analytics
        /// </summary>
        private IList<IActuator> m_Actuators;

        /// <summary>
        /// Whether or not we've tried to send analytics for this model. We only ever try to send once per policy,
        /// and do additional deduplication in the analytics code.
        /// </summary>
        private bool m_AnalyticsSent;

        /// <summary>
        /// Instantiate a BarracudaPolicy with the necessary objects for it to run.
        /// </summary>
        /// <param name="actionSpec">The action spec of the behavior.</param>
        /// <param name="actuators">The actuators used for this behavior.</param>
        /// <param name="model">The Neural Network to use.</param>
        /// <param name="inferenceDevice">Which device Barracuda will run on.</param>
        /// <param name="behaviorName">The name of the behavior.</param>
        public BarracudaPolicy(
            ActionSpec actionSpec,
            IList<IActuator> actuators,
            NNModel model,
            InferenceDevice inferenceDevice,
            string behaviorName
        )
        {
            var modelRunner = Academy.Instance.GetOrCreateModelRunner(model, actionSpec, inferenceDevice);
            m_ModelRunner = modelRunner;
            m_BehaviorName = behaviorName;
            m_ActionSpec = actionSpec;
            m_Actuators = actuators;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            SendAnalytics(sensors);
            m_AgentId = info.episodeId;
            m_ModelRunner?.PutObservations(info, sensors);
        }

        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        void SendAnalytics(IList<ISensor> sensors)
        {
            if (!m_AnalyticsSent)
            {
                m_AnalyticsSent = true;
                Analytics.InferenceAnalytics.InferenceModelSet(
                    m_ModelRunner.Model,
                    m_BehaviorName,
                    m_ModelRunner.InferenceDevice,
                    sensors,
                    m_ActionSpec,
                    m_Actuators
                );
            }
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
