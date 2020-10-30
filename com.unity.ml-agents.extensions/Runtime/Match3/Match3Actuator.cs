using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;


namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Actuator for a Match3 game. It translates valid moves (defined by AbstractBoard.IsMoveValid())
    /// in action masks, and applies the action to the board via AbstractBoard.MakeMove().
    /// </summary>
    public class Match3Actuator : IActuator
    {
        private AbstractBoard m_Board;
        private ActionSpec m_ActionSpec;
        private bool m_ForceHeuristic;
        private System.Random m_Random;
        private Agent m_Agent;

        private int m_Rows;
        private int m_Columns;
        private int m_NumCellTypes;

        /// <summary>
        /// Create a Match3Actuator.
        /// </summary>
        /// <param name="board"></param>
        /// <param name="forceHeuristic">Whether the inference action should be ignored and the Agent's Heuristic
        /// should be called. This should only be used for generating comparison stats of the Heuristic.</param>
        /// <param name="agent"></param>
        /// <param name="name"></param>
        public Match3Actuator(AbstractBoard board, bool forceHeuristic, Agent agent, string name)
        {
            m_Board = board;
            m_Rows = board.Rows;
            m_Columns = board.Columns;
            m_NumCellTypes = board.NumCellTypes;
            Name = name;

            m_ForceHeuristic = forceHeuristic;
            m_Agent = agent;

            var numMoves = Move.NumPotentialMoves(m_Board.Rows, m_Board.Columns);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
        }

        /// <inheritdoc/>
        public ActionSpec ActionSpec => m_ActionSpec;

        /// <inheritdoc/>
        public void OnActionReceived(ActionBuffers actions)
        {
            if (m_ForceHeuristic)
            {
                m_Agent.Heuristic(actions);
            }
            var moveIndex = actions.DiscreteActions[0];

            if (m_Board.Rows != m_Rows || m_Board.Columns != m_Columns || m_Board.NumCellTypes != m_NumCellTypes)
            {
                Debug.LogWarning(
                    $"Board shape changes since actuator initialization. This may cause unexpected results. " +
                    $"Old shape: Rows={m_Rows} Columns={m_Columns}, NumCellTypes={m_NumCellTypes} " +
                    $"Current shape: Rows={m_Board.Rows} Columns={m_Board.Columns}, NumCellTypes={m_Board.NumCellTypes}"
                );
            }

            Move move = Move.FromMoveIndex(moveIndex, m_Rows, m_Columns);
            m_Board.MakeMove(move);
        }

        /// <inheritdoc/>
        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            using (TimerStack.Instance.Scoped("WriteDiscreteActionMask"))
            {
                actionMask.WriteMask(0, InvalidMoveIndices());
            }
        }

        /// <inheritdoc/>
        public string Name { get; }

        /// <inheritdoc/>
        public void ResetData()
        {
        }

        IEnumerable<int> InvalidMoveIndices()
        {
            var numValidMoves = m_Board.NumMoves();

            foreach (var move in m_Board.InvalidMoves())
            {
                numValidMoves--;
                if (numValidMoves == 0)
                {
                    // If all the moves are invalid and we mask all the actions out, this will cause an assert
                    // later on in IDiscreteActionMask. Instead, fire a callback to the user if they provided one,
                    // (or log a warning if not) and leave the last action unmasked. This isn't great, but
                    // an invalid move should be easier to handle than an exception..
                    if (m_Board.OnNoValidMovesAction != null)
                    {
                        m_Board.OnNoValidMovesAction();
                    }
                    else
                    {
                        Debug.LogWarning(
                            "No valid moves are available. The last action will be left unmasked, so " +
                            "an invalid move will be passed to AbstractBoard.MakeMove()."
                        );
                    }
                    // This means the last move won't be returned as an invalid index.
                    yield break;
                }
                yield return move.MoveIndex;
            }
        }
    }
}
