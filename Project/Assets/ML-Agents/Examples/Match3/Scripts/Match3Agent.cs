using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    enum State
    {
        Invalid = -1,

        FindMatches = 0,
        ClearMatched = 1,
        Drop = 2,
        FillEmpty = 3,

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

        private System.Random m_Random;
        private const float k_RewardMultiplier = 0.01f;

        void Awake()
        {
            Board = GetComponent<Match3Board>();
            var seed = Board.RandomSeed == -1 ? gameObject.GetInstanceID() : Board.RandomSeed + 1;
            m_Random = new System.Random(seed);
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();

            Board.InitSettled();
            m_CurrentState = State.FindMatches;
            m_TimeUntilMove = MoveTime;
            m_MovesMade = 0;
        }

        private void FixedUpdate()
        {
            if (Academy.Instance.IsCommunicatorOn)
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
                var numMatched = Board.ClearMatchedCells();
                AddReward(k_RewardMultiplier * numMatched);
                Board.DropCells();
                Board.FillFromAbove();
            }

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

            var nextState = State.Invalid;
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
                    var numMatched = Board.ClearMatchedCells();
                    AddReward(k_RewardMultiplier * numMatched);
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
            for (var index = 0; index < Move.NumPotentialMoves(Board.Rows, Board.Columns); index++)
            {
                var move = Move.FromMoveIndex(index, Board.Rows, Board.Columns);
                if (Board.IsMoveValid(move))
                {
                    return true;
                }
            }

            return false;
        }


        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = Board.GetRandomValidMoveIndex(m_Random);
        }
    }

}
