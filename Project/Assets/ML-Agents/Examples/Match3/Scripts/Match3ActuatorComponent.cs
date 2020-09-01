using System.Runtime.InteropServices;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgentsExamples
{
    public class Match3Actuator : IActuator
    {
        private Match3Agent m_Agent;
        private ActionSpec m_ActionSpec;

        public Match3Actuator(Match3Agent agent)
        {
            m_Agent = agent;
            var numMoves = Move.NumEdgeIndices(m_Agent.Rows, m_Agent.Cols);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
        }

        public ActionSpec ActionSpec => m_ActionSpec;

        public void OnActionReceived(ActionBuffers actions)
        {
            int moveIndex = actions.DiscreteActions[0];
            Move move = Move.FromEdgeIndex(moveIndex, m_Agent.Rows, m_Agent.Cols);
            m_Agent.Board.MakeMove(move);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {

        }

        public string Name => "Match3Actuator";

        public void ResetData()
        {

        }
    }

    public class Match3ActuatorComponent : ActuatorComponent
    {
        public Match3Agent Agent;
        public override IActuator CreateActuator()
        {
            return new Match3Actuator(Agent);
        }

        public override ActionSpec ActionSpec
        {
            get
            {
                if (Agent == null)
                {
                    return ActionSpec.MakeContinuous(0);
                }

                var numMoves = Move.NumEdgeIndices(Agent.Rows, Agent.Cols);
                return ActionSpec.MakeDiscrete(numMoves);
            }
        }
    }
}
