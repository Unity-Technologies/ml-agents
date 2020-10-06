using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;


namespace Unity.MLAgents.Extensions.Match3
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
            var numMoves = Move.NumPotentialMoves(m_Board.Rows, m_Board.Columns);
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
            Move move = Move.FromMoveIndex(moveIndex, m_Board.Rows, m_Board.Columns);
            m_Board.MakeMove(move);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            using (TimerStack.Instance.Scoped("WriteDiscreteActionMask"))
            {
                actionMask.WriteMask(0, InvalidMoveIndices());
            }
        }

        public string Name => "Match3Actuator";// TODO pass optional name

        public void ResetData()
        {
        }

        IEnumerable<int> InvalidMoveIndices()
        {
            foreach (var move in m_Board.InvalidMoves())
            {
                yield return move.InternalEdgeIndex;
            }
        }
    }
}
