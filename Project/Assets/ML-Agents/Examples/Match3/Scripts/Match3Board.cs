using System;
using UnityEngine;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{


    public class Match3Board : AbstractBoard
    {
        const int k_EmptyCell = -1;

        readonly int[,] m_Cells;
        readonly bool[,] m_Matched;

        System.Random m_Random;

        public Match3Board(int rows, int cols, int numCellTypes, int randomSeed) : base(rows, cols, numCellTypes)
        {
            m_Cells = new int[cols, rows];
            m_Matched = new bool[cols, rows];

            m_Random = new System.Random(randomSeed);

            InitRandom();
        }

        public override bool MakeMove(Move move)
        {
            if (!IsMoveValid(move))
            {
                return false;
            }
            var originalValue = m_Cells[move.m_Column, move.m_Row];
            var (otherRow, otherCol) = move.OtherCell();
            var destinationValue = m_Cells[otherCol, otherRow];

            m_Cells[move.m_Column, move.m_Row] = destinationValue;
            m_Cells[otherCol, otherRow] = originalValue;
            return true;
        }

        public override int GetCellType(int row, int col)
        {
            return m_Cells[col, row];
        }

        public override bool IsMoveValid(Move move)
        {
            var moveVal = m_Cells[move.m_Column, move.m_Row];
            var (otherRow, otherCol) = move.OtherCell();
            var oppositeVal = m_Cells[otherCol, otherRow];

            // Simple check - if the values are the same, don't match
            // This might not be valid for all games
            {
                if (moveVal == oppositeVal)
                {
                    return false;
                }
            }

            bool moveMatches = CheckHalfMove(otherRow, otherCol, m_Cells[move.m_Column, move.m_Row], move.m_Direction);
            bool otherMatches = CheckHalfMove(move.m_Row, move.m_Column, m_Cells[otherCol, otherRow],
                move.OtherDirection());

            return moveMatches || otherMatches;
        }

        bool CheckHalfMove(int newRow, int newCol, int newValue, Direction incomingDirection)
        {
            int matchedLeft = 0, matchedRight = 0, matchedUp = 0, matchedDown = 0;

            if (incomingDirection != Direction.Right)
            {
                for (var c = newCol - 1; c >= 0; c--)
                {
                    if (m_Cells[c, newRow] == newValue)
                        matchedLeft++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Left)
            {
                for (var c = newCol + 1; c < Columns; c++)
                {
                    if (m_Cells[c, newRow] == newValue)
                        matchedRight++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Down)
            {
                for (var r = newRow + 1; r < Rows; r++)
                {
                    if (m_Cells[newCol, r] == newValue)
                        matchedUp++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Up)
            {
                for (var r = newRow - 1; r >= 0; r--)
                {
                    if (m_Cells[newCol, r] == newValue)
                        matchedDown++;
                    else
                        break;
                }
            }

            if ((matchedUp + matchedDown >= 2) || (matchedLeft + matchedRight >= 2))
            {
                return true;
            }

            return false;
        }

        public bool MarkMatchedCells(int[,] cells = null)
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

        public int ClearMatchedCells()
        {
            int numMatchedCells = 0;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Matched[j, i])
                    {
                        numMatchedCells++;
                        m_Cells[j, i] = k_EmptyCell;
                    }
                }
            }

            ClearMarked(); // TODO clear here or at start of matching?
            return numMatchedCells;
        }

        public bool DropCells()
        {
            var madeChanges = false;
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
                    madeChanges = true;
                    m_Cells[j, writeIndex] = k_EmptyCell;
                }
            }

            return madeChanges;
        }

        public bool FillFromAbove()
        {
            bool madeChanges = false;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Cells[j, i] == k_EmptyCell)
                    {
                        madeChanges = true;
                        m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
                    }
                }
            }

            return madeChanges;
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
        public void InitRandom()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
                }
            }
        }

        public void InitSettled()
        {
            InitRandom();
            while (true)
            {
                var anyMatched = MarkMatchedCells();
                if (!anyMatched)
                {
                    return;
                }
                ClearMatchedCells();
                DropCells();
                FillFromAbove();
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
