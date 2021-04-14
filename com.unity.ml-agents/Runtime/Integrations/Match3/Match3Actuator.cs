using Unity.MLAgents.Actuators;
using Debug = UnityEngine.Debug;


namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Actuator for a Match3 game. It translates valid moves (defined by AbstractBoard.IsMoveValid())
    /// in action masks, and applies the action to the board via AbstractBoard.MakeMove().
    /// </summary>
    public class Match3Actuator : IActuator, IBuiltInActuator
    {
        AbstractBoard m_Board;
        System.Random m_Random;
        ActionSpec m_ActionSpec;
        bool m_ForceHeuristic;
        BoardSize m_MaxBoardSize;

        /// <summary>
        /// Create a Match3Actuator.
        /// </summary>
        /// <param name="board"></param>
        /// <param name="forceHeuristic">Whether the inference action should be ignored and the Agent's Heuristic
        /// should be called. This should only be used for generating comparison stats of the Heuristic.</param>
        /// <param name="seed">The seed used to initialize <see cref="System.Random"/>.</param>
        /// <param name="name"></param>
        public Match3Actuator(AbstractBoard board,
                              bool forceHeuristic,
                              int seed,
                              string name)
        {
            m_Board = board;
            m_MaxBoardSize = m_Board.GetMaxBoardSize();
            Name = name;

            m_ForceHeuristic = forceHeuristic;

            var numMoves = Move.NumPotentialMoves(m_MaxBoardSize);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
            m_Random = new System.Random(seed);
        }

        /// <inheritdoc/>
        public ActionSpec ActionSpec => m_ActionSpec;

        /// <inheritdoc/>
        public void OnActionReceived(ActionBuffers actions)
        {
            m_Board.CheckBoardSizes(m_MaxBoardSize);
            if (m_ForceHeuristic)
            {
                Heuristic(actions);
            }
            var moveIndex = actions.DiscreteActions[0];

            Move move = Move.FromMoveIndex(moveIndex, m_MaxBoardSize);
            m_Board.MakeMove(move);
        }

        /// <inheritdoc/>
        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            var currentBoardSize = m_Board.GetCurrentBoardSize();
            m_Board.CheckBoardSizes(m_MaxBoardSize);
            const int branch = 0;
            bool foundValidMove = false;
            using (TimerStack.Instance.Scoped("WriteDiscreteActionMask"))
            {
                var numMoves = m_Board.NumMoves();

                var currentMove = Move.FromMoveIndex(0, m_MaxBoardSize);
                for (var i = 0; i < numMoves; i++)
                {
                    // Check that the move is allowed for the current boardSize (e.g. it won't move a piece out of
                    // bounds), and that it's allowed by the game itself.
                    if (currentMove.InRangeForBoard(currentBoardSize) && m_Board.IsMoveValid(currentMove))
                    {
                        foundValidMove = true;
                    }
                    else
                    {
                        actionMask.SetActionEnabled(branch, i, false);
                    }
                    currentMove.Next(m_MaxBoardSize);
                }

                if (!foundValidMove)
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
                    actionMask.SetActionEnabled(branch, numMoves - 1, true);
                }
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

        /// <inheritdoc/>
        public void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = GreedyMove();
        }

        /// <summary>
        /// Returns a valid move that gives the highest value for EvalMovePoints(). If multiple moves have the same
        /// value, one of them will be chosen with uniform probability.
        /// </summary>
        /// <remarks>
        /// By default, EvalMovePoints() returns 1, so all valid moves are equally likely. Inherit from this class and
        /// override EvalMovePoints() to use your game's scoring as a better estimate.
        /// </remarks>
        /// <returns></returns>
        internal int GreedyMove()
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
