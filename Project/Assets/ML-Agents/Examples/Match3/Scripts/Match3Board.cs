using System;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public enum Direction
    {
        Up, // +row direction
        Down, // -row direction
        Left, // -column direction
        Right, // +column direction
    }

    /// <summary>
    /// Struct that encapsulates a swpa of adjacent cells.
    /// A Move can be constructed from either a starting cells and a direction,
    /// or enumerated from 0 to NumEdgeIndices()-1
    /// </summary>
    public struct Move
    {

        /**
         * Moves are enumerated as the internal edges of the game grid.
         * Left/right moves come first. There are (maxCols - 1) * maxRows of these.
         * Up/down moves are next. There are (maxRows - 1) * maxCols of these.
         */
        public int m_InternalEdgeIndex;
        public int m_Row;
        public int m_Column;
        public Direction m_Direction;

        public static Move FromEdgeIndex(int edgeIndex, int maxRows, int maxCols)
        {
            if (edgeIndex < 0 || edgeIndex >= NumEdgeIndices(maxRows, maxCols))
            {
                throw new ArgumentOutOfRangeException("Invalid edge index.");
            }
            Direction dir;
            int row, col;
            if (edgeIndex < (maxCols - 1) * maxRows)
            {
                dir = Direction.Right;
                col = edgeIndex % (maxCols - 1);
                row = edgeIndex / (maxCols - 1);
            }
            else
            {
                dir = Direction.Up;
                var offset = edgeIndex - (maxCols - 1) * maxRows;
                col = offset % maxCols;
                row = offset / maxCols;
            }
            return new Move
            {
                m_InternalEdgeIndex = edgeIndex,
                m_Direction = dir,
                m_Row = row,
                m_Column = col
            };
        }

        public static Move FromPositionAndDirection(int row, int col, Direction dir, int maxRows, int maxCols)
        {
            int edgeIndex;
            // Normalize - only consider Right and Up
            if (dir == Direction.Left)
            {
                dir = Direction.Right;
                col = col - 1;
            }
            else if (dir == Direction.Down)
            {
                dir = Direction.Up;
                row = row - 1;
            }

            if (dir == Direction.Right)
            {
                edgeIndex = col + row * (maxCols - 1);
            }
            else
            {
                var offset = (maxCols - 1) * maxRows;
                edgeIndex = offset + col + row * maxCols;
            }

            return new Move
            {
                m_Row = row,
                m_Column = col,
                m_Direction = dir,
                m_InternalEdgeIndex = edgeIndex,
            };
        }

        public (int Row, int Column) OtherCell()
        {
            switch (m_Direction)
            {
                case Direction.Up:
                    return (m_Row + 1, m_Column);
                case Direction.Down:
                    return (m_Row - 1, m_Column);
                case Direction.Left:
                    return (m_Row, m_Column - 1);
                case Direction.Right:
                    return (m_Row, m_Column + 1);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public Direction OtherDirection()
        {
            switch (m_Direction)
            {
                case Direction.Up:
                    return Direction.Down;
                case Direction.Down:
                    return Direction.Up;
                case Direction.Left:
                    return Direction.Right;
                case Direction.Right:
                    return Direction.Left;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        /// Return the number of internal edges in a board of the given size.
        /// </summary>
        /// <param name="maxRows"></param>
        /// <param name="maxCols"></param>
        /// <returns></returns>
        public static int NumEdgeIndices(int maxRows, int maxCols)
        {
            return maxRows * (maxCols - 1) + (maxRows - 1) * (maxCols);
        }
    }

    public class Match3Board
    {
        const int k_EmptyCell = -1;

        public readonly int Rows;
        public readonly int Columns;
        public readonly int NumCellTypes;
        readonly int[,] m_Cells;
        readonly int[,] m_TempCells;
        readonly bool[,] m_Matched;

        System.Random m_Random;

        public Match3Board(int rows, int cols, int numCellTypes, int randomSeed)
        {
            Rows = rows;
            Columns = cols;
            NumCellTypes = numCellTypes;
            m_Cells = new int[cols, rows];
            m_TempCells = new int[cols, rows];
            m_Matched = new bool[cols, rows];

            m_Random = new System.Random(randomSeed);

            InitRandom();
            MarkMatchedCells();
        }

        public bool MakeMove(Move move)
        {
            return true;
        }

        public bool IsMoveValid(Move move)
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

            // Check if the cell being moved into (move.m_Column, move.m_Row) completes a 3-match (or better)
            // {
            //     int matchedLeft = 0, matchedRight = 0, matchedUp = 0, matchedDown = 0;
            //     for (var c = move.m_Column - 1; c >= 0; c--)
            //     {
            //         if (m_Cells[c, move.m_Row] == oppositeVal)
            //             matchedLeft++;
            //         else
            //             break;
            //     }
            //
            //     for (var c = move.m_Column + 1; c < Columns; c++)
            //     {
            //         if (m_Cells[c, move.m_Row] == oppositeVal)
            //             matchedRight++;
            //         else
            //             break;
            //     }
            //
            //     for (var r = move.m_Row - 1; r >= 0; r--)
            //     {
            //         if (m_Cells[move.m_Column, r] == oppositeVal)
            //             matchedUp++;
            //         else
            //             break;
            //     }
            //
            //     for (var r = move.m_Row + 1; r < Rows; r++)
            //     {
            //         if (m_Cells[move.m_Column, r] == oppositeVal)
            //             matchedDown++;
            //         else
            //             break;
            //     }
            //
            //     if ((matchedUp + matchedDown >= 2) || (matchedLeft + matchedRight >= 2))
            //     {
            //         return true;
            //     }
            // }
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
