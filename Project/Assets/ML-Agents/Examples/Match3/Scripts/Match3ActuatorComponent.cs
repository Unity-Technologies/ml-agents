using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class Match3Actuator : IActuator
    {
        private AbstractBoard m_Board;
        private ActionSpec m_ActionSpec;
        private bool m_ForceRandom;
        private System.Random m_Random;

        public Match3Actuator(AbstractBoard board, bool forceRandom, int randomSeed)
        {
            m_Board = board;
            m_ForceRandom = forceRandom;
            if (forceRandom)
            {
                m_Random = new System.Random(randomSeed);
            }
            var numMoves = Move.NumEdgeIndices(m_Board.Rows, m_Board.Columns);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
        }

        public ActionSpec ActionSpec => m_ActionSpec;

        public void OnActionReceived(ActionBuffers actions)
        {
            int moveIndex = 0;
            if (m_ForceRandom)
            {
                moveIndex = m_Board.GetRandomValidMoveIndex(m_Random);
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

        public string Name => "Match3Actuator";// TODO pass optional name

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
            var randomSeed = 0;
            if (ForceRandom)
            {
                randomSeed = this.gameObject.GetInstanceID();
            }
            return new Match3Actuator(Agent.Board, ForceRandom, randomSeed);
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
