using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3Actuator : IActuator
    {
        private AbstractBoard m_Board;
        private ActionSpec m_ActionSpec;
        private bool m_ForceRandom;

        public Match3Actuator(AbstractBoard board, bool forceRandom)
        {
            m_Board = board;
            m_ForceRandom = forceRandom;
            var numMoves = Move.NumEdgeIndices(m_Board.Rows, m_Board.Columns);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
        }

        public ActionSpec ActionSpec => m_ActionSpec;

        public void OnActionReceived(ActionBuffers actions)
        {
            int moveIndex;
            if (m_ForceRandom)
            {
                //moveIndex = m_Agent.GetRandomValidMoveIndex(); TODO
            }
            else
            {
                moveIndex = actions.DiscreteActions[0];
            }
            Move move = Move.FromEdgeIndex(moveIndex, m_Board.Rows, m_Board.Columns);
            m_Board.MakeMove(move);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            actionMask.WriteMask(0, InvalidMoveIndices());
        }

        public string Name => "Match3Actuator";

        public void ResetData()
        {

        }

        IEnumerable<int> InvalidMoveIndices()
        {
            var numMoves = Move.NumEdgeIndices(m_Board.Rows, m_Board.Columns);
            for (var i = 0; i < numMoves; i++)
            {
                Move move = Move.FromEdgeIndex(i, m_Board.Rows, m_Board.Columns);
                if (!m_Board.IsMoveValid(move))
                {
                    yield return i;
                }
            }
        }
    }

    public class Match3ActuatorComponent : ActuatorComponent
    {
        public Match3Agent Agent;
        public bool ForceRandom = false;
        public override IActuator CreateActuator()
        {
            return new Match3Actuator(Agent.Board, ForceRandom);
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
