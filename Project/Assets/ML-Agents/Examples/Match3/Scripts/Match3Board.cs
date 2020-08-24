using System;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public enum Direction
    {
        Up,
        Down,
        Left,
        Right,
    }

    public class Match3Board
    {
        const int k_EmptyCell = -1;

        public readonly int Rows;
        public readonly int Columns;
        public readonly int NumCellTypes;
        readonly int[,] m_Cells;
        readonly bool[,] m_Matched;

        System.Random m_Random;

        public Match3Board(int rows, int cols, int numCellTypes, int randomSeed)
        {
            Rows = rows;
            Columns = cols;
            NumCellTypes = numCellTypes;
            m_Cells = new int[cols, rows];
            m_Matched = new bool[cols, rows];

            m_Random = new System.Random(randomSeed);

            InitRandom();
            MarkMatchedCells();
        }

        public bool MakeMove(int row, int col, Direction dir)
        {
            return true;
        }

        public bool IsMoveValid(int row, int col, Direction dir)
        {
            return true;
        }

        public bool MarkMatchedCells()
        {
            ClearMarked();
            bool madeMatch = false;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    // Check vertically
                    var matchedRows = 0;
                    for (var iOffset = i; iOffset < Rows; iOffset++)
                    {
                        if (m_Cells[j, i] != m_Cells[j, iOffset])
                        {
                            break;
                        }

                        matchedRows++;
                    }

                    if (matchedRows >= 3)
                    {
                        madeMatch = true;
                        for (var k = 0; k < matchedRows; k++)
                        {
                            // TODO check whether already matched for scoring
                            m_Matched[j, i + k] = true;
                        }
                    }

                    // Check vertically
                    var matchedCols = 0;
                    for (var jOffset = j; jOffset < Columns; jOffset++)
                    {
                        if (m_Cells[j, i] != m_Cells[jOffset, i])
                        {
                            break;
                        }

                        matchedCols++;
                    }

                    if (matchedCols >= 3)
                    {
                        madeMatch = true;
                        for (var k = 0; k < matchedCols; k++)
                        {
                            // TODO check whether already matched for scoring
                            m_Matched[j + k, i] = true;
                        }
                    }
                }
            }

            return madeMatch;
        }

        public bool ClearMatchedCells()
        {
            bool hasMatchedCell = false;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Matched[j, i])
                    {
                        hasMatchedCell = true;
                        m_Cells[j, i] = k_EmptyCell;
                    }
                }
            }

            ClearMarked(); // TODO clear here or at start of matching?
            return hasMatchedCell;
        }

        public void DropCells()
        {
            // Gravity is applied in the negative row direction
            for (var j = 0; j < Columns; j++)
            {
                var writeIndex = 0;
                for (var readIndex = 0; readIndex < Rows; readIndex++)
                {
                    m_Cells[j, writeIndex] = m_Cells[j, readIndex];
                    if (m_Cells[j, readIndex] != k_EmptyCell)
                    {
                        writeIndex++;
                    }
                }

                // Fill in empties at the end
                // TODO combine with random drops?
                for (; writeIndex < Rows; writeIndex++)
                {
                    m_Cells[j, writeIndex] = k_EmptyCell;
                }
            }
        }

        public void FillFromAbove()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Cells[j, i] == k_EmptyCell)
                    {
                        m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
                    }
                }
            }
        }

        public int[,] Cells
        {
            get { return m_Cells; }
        }

        public bool[,] Matched
        {
            get { return m_Matched; }
        }

        // Initialize the board to random values.
        void InitRandom()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
                }
            }
        }

        void ClearMarked()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m_Matched[j, i] = false;
                }
            }
        }


    }
}
