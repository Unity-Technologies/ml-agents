// Policy for C# training

using Unity.Barracuda;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    internal class TrainingPolicy : IPolicy
    {
        protected TrainingModelRunner m_ModelRunner;
        ActionBuffers m_LastActionBuffer;

        int m_AgentId;

        ActionSpec m_ActionSpec;

        private string m_BehaviorName;

        /// <inheritdoc />
        public TrainingPolicy(
            ActionSpec actionSpec,
            string behaviorName
        )
        {
            m_ModelRunner = Academy.Instance.GetOrCreateTrainingModelRunner(behaviorName, actionSpec);
            m_BehaviorName = behaviorName;
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
