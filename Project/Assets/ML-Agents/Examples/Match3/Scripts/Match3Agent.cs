using System;
using System.Security.Principal;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;

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

        NumSteps = 4
    }

    public class Match3Agent : Agent
    {
        [HideInInspector]
        public Match3Board Board;
        public int Rows = 8;
        public int Cols = 8;
        public int NumCellTypes = 6;
        public int RandomSeed = 1337;

        public float MoveTime = 1.0f;

        State m_CurrentState = State.FindMatches;
        float m_TimeUntilMove;

        private bool[] m_ValidMoves;
        private System.Random m_Random;
        private const float k_RewardMultiplier = 0.01f;

        void Awake()
        {
            Board = new Match3Board(Rows, Cols, NumCellTypes, RandomSeed);
            m_ValidMoves = new bool[Move.NumEdgeIndices(Rows, Cols)];
            m_Random = new System.Random(RandomSeed + 1);
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();

            Board.InitRandom();
            m_CurrentState = State.FindMatches;
            m_TimeUntilMove = MoveTime;
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

            bool hasMoves = CheckValidMoves();
            if (hasMoves)
            {
                RequestDecision();
            }
            else
            {
                EndEpisode();
            }
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
                    bool hasMoves = CheckValidMoves();
                    if (hasMoves)
                    {
                        RequestDecision();
                    }
                    else
                    {
                        EndEpisode();
                    }

                    nextState = State.FindMatches;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            m_CurrentState = nextState;
        }

        bool CheckValidMoves()
        {
            int numValidMoves = 0;
            Array.Clear(m_ValidMoves, 0, m_ValidMoves.Length);

            for (var index = 0; index < Move.NumEdgeIndices(Rows, Cols); index++)
            {
                var move = Move.FromEdgeIndex(index, Rows, Cols);
                if (Board.IsMoveValid(move))
                {
                    m_ValidMoves[index] = true;
                    numValidMoves++;
                }
            }

            return numValidMoves > 0;
        }


        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = GetRandomValidMoveIndex();
        }

        public int GetRandomValidMoveIndex()
        {
            // Pick a random valid move
            // TODO reservoir sample?
            var numValidMoves = 0;
            foreach (var valid in m_ValidMoves)
            {
                numValidMoves += valid ? 1 : 0;
            }

            if (numValidMoves == 0)
            {
                Debug.Log("No valid moves");
                return -1;
            }

            // We'll make the n'th valid move where n in [0, numValidMoves)
            var target = m_Random.Next(numValidMoves);
            var numSkipped = 0;

            for (var i = 0; i < m_ValidMoves.Length; i++)
            {
                var valid = m_ValidMoves[i];
                if (valid)
                {
                    if (numSkipped == target)
                    {
                        return i;
                    }

                    numSkipped++;
                }
            }

            // Should never reach here
            return -1;
        }
    }

}
