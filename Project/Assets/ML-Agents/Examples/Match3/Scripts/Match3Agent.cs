using System;
using Unity.MLAgents;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    enum Steps
    {
        FindMatches = 0,
        ClearMatched = 1,
        Drop = 2,
        FillEmpty = 3,

        NumSteps = 4
    }

    public class Match3Agent : Agent
    {
        [HideInInspector]
        public Match3Board Board;
        int Rows = 8;
        int Cols = 8;
        int NumCellTypes = 6;

        public float MoveTime = 5.0f;

        Steps m_Current = Steps.FindMatches;
        float m_TimeUntilMove;


        void Awake()
        {
            Board = new Match3Board(Rows, Cols, NumCellTypes, 1337);
            m_TimeUntilMove = MoveTime;
        }

        void Update()
        {
            m_TimeUntilMove -= Time.deltaTime;
            if (m_TimeUntilMove > 0.0f)
            {
                return;
            }

            m_TimeUntilMove = MoveTime;

            switch (m_Current)
            {
                case Steps.FindMatches:
                    Board.MarkMatchedCells();
                    break;
                case Steps.ClearMatched:
                    Board.ClearMatchedCells();
                    break;
                case Steps.Drop:
                    Board.DropCells();
                    break;
                case Steps.FillEmpty:
                    Board.FillFromAbove();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            m_Current = (Steps)(m_Current + 1);
            if (m_Current == Steps.NumSteps)
            {
                m_Current = (Steps)0;
            }

        }

    }
}
