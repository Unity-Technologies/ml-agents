using System;
using UnityEngine;
using Unity.MLAgents;

namespace Unity.MLAgentsExamples
{

    /// <summary>
    /// State of the "game" when showing all steps of the simulation. This is only used outside of training.
    /// The state diagram is
    ///
    ///      | <--------------------------------------- ^
    ///      |                                          |
    ///      v                                          |
    ///  +--------+      +-------+      +-----+      +------+
    ///  |Find    | ---> |Clear  | ---> |Drop | ---> |Fill  |
    ///  |Matches |      |Matched|      |     |      |Empty |
    ///  +--------+      +-------+      +-----+      +------+
    ///
    ///    |     ^
    ///    |     |
    ///    v     |
    ///
    ///  +--------+
    ///  |Wait for|
    ///  |Move    |
    ///  +--------+
    ///
    /// The stats advances each "MoveTime" seconds.
    /// </summary>
    enum State
    {
        /// <summary>
        /// Guard value, should never happen.
        /// </summary>
        Invalid = -1,

        /// <summary>
        /// Look for matches. If there are matches, the next state is ClearMatched, otherwise WaitForMove.
        /// </summary>
        FindMatches = 0,

        /// <summary>
        /// Remove matched cells and replace them with a placeholder value.
        /// </summary>
        ClearMatched = 1,

        /// <summary>
        /// Move cells "down" to fill empty space.
        /// </summary>
        Drop = 2,

        /// <summary>
        /// Replace empty cells with new random values.
        /// </summary>
        FillEmpty = 3,

        /// <summary>
        /// Request a move from the Agent.
        /// </summary>
        WaitForMove = 4,
    }

    public class Match3Agent : Agent
    {
        [HideInInspector]
        public Match3Board Board;

        public float MoveTime = 1.0f;
        public int MaxMoves = 500;


        State m_CurrentState = State.WaitForMove;
        float m_TimeUntilMove;
        private int m_MovesMade;
        private ModelOverrider m_ModelOverrider;

        private const float k_RewardMultiplier = 0.01f;
        void Awake()
        {
            Board = GetComponent<Match3Board>();
            m_ModelOverrider = GetComponent<ModelOverrider>();
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();

            Board.UpdateCurrentBoardSize();
            Board.InitSettled();
            m_CurrentState = State.FindMatches;
            m_TimeUntilMove = MoveTime;
            m_MovesMade = 0;
        }

        private void FixedUpdate()
        {
            // Make a move every step if we're training, or we're overriding models in CI.
            var useFast = Academy.Instance.IsCommunicatorOn || (m_ModelOverrider != null && m_ModelOverrider.HasOverrides);
            if (useFast)
            {
                FastUpdate();
            }
            else
            {
                AnimatedUpdate();
            }

            // We can't use the normal MaxSteps system to decide when to end an episode,
            // since different agents will make moves at different frequencies (depending on the number of
            // chained moves). So track a number of moves per Agent and manually interrupt the episode.
            if (m_MovesMade >= MaxMoves)
            {
                EpisodeInterrupted();
            }
        }

        void FastUpdate()
        {
            while (true)
            {
                var hasMatched = Board.MarkMatchedCells();
                if (!hasMatched)
                {
                    break;
                }
                var pointsEarned = Board.ClearMatchedCells();
                AddReward(k_RewardMultiplier * pointsEarned);
                Board.DropCells();
                Board.FillFromAbove();
            }

            while (!HasValidMoves())
            {
                // Shuffle the board until we have a valid move.
                Board.InitSettled();
            }
            RequestDecision();
            m_MovesMade++;
        }

        void AnimatedUpdate()
        {
            m_TimeUntilMove -= Time.deltaTime;
            if (m_TimeUntilMove > 0.0f)
            {
                return;
            }

            m_TimeUntilMove = MoveTime;

            State nextState;
            switch (m_CurrentState)
            {
                case State.FindMatches:
                    var hasMatched = Board.MarkMatchedCells();
                    nextState = hasMatched ? State.ClearMatched : State.WaitForMove;
                    if (nextState == State.WaitForMove)
                    {
                        m_MovesMade++;
                    }
                    break;
                case State.ClearMatched:
                    var pointsEarned = Board.ClearMatchedCells();
                    AddReward(k_RewardMultiplier * pointsEarned);
                    nextState = State.Drop;
                    break;
                case State.Drop:
                    Board.DropCells();
                    nextState = State.FillEmpty;
                    break;
                case State.FillEmpty:
                    Board.FillFromAbove();
                    nextState = State.FindMatches;
                    break;
                case State.WaitForMove:
                    while (true)
                    {
                        // Shuffle the board until we have a valid move.
                        bool hasMoves = HasValidMoves();
                        if (hasMoves)
                        {
                            break;
                        }
                        Board.InitSettled();
                    }
                    RequestDecision();

                    nextState = State.FindMatches;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            m_CurrentState = nextState;
        }

        bool HasValidMoves()
        {
            foreach (var unused in Board.ValidMoves())
            {
                return true;
            }

            return false;
        }

    }

}
