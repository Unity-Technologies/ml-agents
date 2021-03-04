using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;


namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Actuator for a Match3 game. It translates valid moves (defined by AbstractBoard.IsMoveValid())
    /// in action masks, and applies the action to the board via AbstractBoard.MakeMove().
    /// </summary>
    public class Match3Actuator : IActuator, IHeuristicProvider, IBuiltInActuator
    {
        protected AbstractBoard m_Board;
        protected System.Random m_Random;
        private ActionSpec m_ActionSpec;
        private bool m_ForceHeuristic;
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
        /// <param name="seed">The seed used to initialize <see cref="System.Random"/>.</param>
        /// <param name="agent"></param>
        /// <param name="name"></param>
        public Match3Actuator(AbstractBoard board,
            bool forceHeuristic,
            int seed,
            Agent agent,
            string name)
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
            m_Random = new System.Random(seed);
        }

        /// <inheritdoc/>
        public ActionSpec ActionSpec => m_ActionSpec;

        /// <inheritdoc/>
        public void OnActionReceived(ActionBuffers actions)
        {
            if (m_ForceHeuristic)
            {
                Heuristic(actions);
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

        /// <inheritdoc/>
        public BuiltInActuatorType GetBuiltInActuatorType()
        {
            return BuiltInActuatorType.Match3Actuator;
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

        public void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = GreedyMove();
        }


        protected int GreedyMove()
        {

            var bestMoveIndex = 0;
            var bestMovePoints = -1;
            var numMovesAtCurrentScore = 0;

            foreach (var move in m_Board.ValidMoves())
            {
                var movePoints = EvalMovePoints(move);
                if (movePoints < bestMovePoints)
                {
                    // Worse, skip
                    continue;
                }

                if (movePoints > bestMovePoints)
                {
                    // Better, keep
                    bestMovePoints = movePoints;
                    bestMoveIndex = move.MoveIndex;
                    numMovesAtCurrentScore = 1;
                }
                else
                {
                    // Tied for best - use reservoir sampling to make sure we select from equal moves uniformly.
                    // See https://en.wikipedia.org/wiki/Reservoir_sampling#Simple_algorithm
                    numMovesAtCurrentScore++;
                    var randVal = m_Random.Next(0, numMovesAtCurrentScore);
                    if (randVal == 0)
                    {
                        // Keep the new one
                        bestMoveIndex = move.MoveIndex;
                    }
                }
            }

            return bestMoveIndex;
        }

        /// <summary>
        /// Method to be overridden when evaluating how many points a specific move will generate.
        /// </summary>
        /// <param name="move">The move to evaluate.</param>
        /// <returns>The number of points the move generates.</returns>
        protected virtual int EvalMovePoints(Move move)
        {
            return 1;
        }
    }
}
