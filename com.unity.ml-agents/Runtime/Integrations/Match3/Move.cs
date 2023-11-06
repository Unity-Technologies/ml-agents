using System;
using UnityEngine;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Directions for a Move.
    /// </summary>
    public enum Direction
    {
        /// <summary>
        /// Move up (increasing row direction).
        /// </summary>
        Up,

        /// <summary>
        /// Move down (decreasing row direction).
        /// </summary>
        Down, // -row direction

        /// <summary>
        /// Move left (decreasing column direction).
        /// </summary>
        Left, // -column direction

        /// <summary>
        /// Move right (increasing column direction).
        /// </summary>
        Right, // +column direction
    }

    /// <summary>
    /// Struct that encapsulates a swap of adjacent cells.
    /// A Move can be constructed from either a starting row, column, and direction,
    /// or from a "move index" between 0 and NumPotentialMoves()-1.
    /// Moves are enumerated as the internal edges of the game grid.
    /// Left/right moves come first. There are (maxCols - 1) * maxRows of these.
    /// Up/down moves are next. There are (maxRows - 1) * maxCols of these.
    /// </summary>
    public struct Move
    {
        /// <summary>
        /// Index of the move, from 0 to NumPotentialMoves-1.
        /// </summary>
        public int MoveIndex;

        /// <summary>
        /// Row of the cell that will be moved.
        /// </summary>
        public int Row;

        /// <summary>
        /// Column of the cell that will be moved.
        /// </summary>
        public int Column;

        /// <summary>
        /// Direction that the cell will be moved.
        /// </summary>
        public Direction Direction;

        /// <summary>
        /// Construct a Move from its move index and the board size.
        /// This is useful for iterating through all the Moves on a board, or constructing
        /// the Move corresponding to an Agent decision.
        /// </summary>
        /// <param name="moveIndex">Must be between 0 and NumPotentialMoves(maxRows, maxCols).</param>
        /// <param name="maxBoardSize"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static Move FromMoveIndex(int moveIndex, BoardSize maxBoardSize)
        {
            var maxRows = maxBoardSize.Rows;
            var maxCols = maxBoardSize.Columns;

            if (moveIndex < 0 || moveIndex >= NumPotentialMoves(maxBoardSize))
            {
                throw new ArgumentOutOfRangeException("moveIndex");
            }
            Direction dir;
            int row, col;
            if (moveIndex < (maxCols - 1) * maxRows)
            {
                dir = Direction.Right;
                col = moveIndex % (maxCols - 1);
                row = moveIndex / (maxCols - 1);
            }
            else
            {
                dir = Direction.Up;
                var offset = moveIndex - (maxCols - 1) * maxRows;
                col = offset % maxCols;
                row = offset / maxCols;
            }
            return new Move
            {
                MoveIndex = moveIndex,
                Direction = dir,
                Row = row,
                Column = col
            };
        }

        /// <summary>
        /// Increment the Move to the next MoveIndex, and update the Row, Column, and Direction accordingly.
        /// </summary>
        /// <param name="maxBoardSize"></param>
        public void Next(BoardSize maxBoardSize)
        {
            var maxRows = maxBoardSize.Rows;
            var maxCols = maxBoardSize.Columns;

            var switchoverIndex = (maxCols - 1) * maxRows;

            MoveIndex++;
            if (MoveIndex < switchoverIndex)
            {
                Column++;
                if (Column == maxCols - 1)
                {
                    Row++;
                    Column = 0;
                }
            }
            else if (MoveIndex == switchoverIndex)
            {
                // switch from moving right to moving up
                Row = 0;
                Column = 0;
                Direction = Direction.Up;
            }
            else
            {
                Column++;
                if (Column == maxCols)
                {
                    Row++;
                    Column = 0;
                }
            }
        }

        /// <summary>
        /// Construct a Move from the row, column, direction, and board size.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="dir"></param>
        /// <param name="maxBoardSize"></param>
        /// <returns></returns>
        public static Move FromPositionAndDirection(int row, int col, Direction dir, BoardSize maxBoardSize)
        {
            // Check for out-of-bounds
            if (row < 0 || row >= maxBoardSize.Rows)
            {
                throw new IndexOutOfRangeException($"row was {row}, but must be between 0 and {maxBoardSize.Rows - 1}.");
            }

            if (col < 0 || col >= maxBoardSize.Columns)
            {
                throw new IndexOutOfRangeException($"col was {col}, but must be between 0 and {maxBoardSize.Columns - 1}.");
            }

            // Check moves that would go out of bounds e.g. col == 0 and dir == Left
            if (
                row == 0 && dir == Direction.Down ||
                row == maxBoardSize.Rows - 1 && dir == Direction.Up ||
                col == 0 && dir == Direction.Left ||
                col == maxBoardSize.Columns - 1 && dir == Direction.Right
            )
            {
                throw new IndexOutOfRangeException($"Cannot move cell at row={row} col={col} in Direction={dir}");
            }

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

            int moveIndex;
            if (dir == Direction.Right)
            {
                moveIndex = col + row * (maxBoardSize.Columns - 1);
            }
            else
            {
                var offset = (maxBoardSize.Columns - 1) * maxBoardSize.Rows;
                moveIndex = offset + col + row * maxBoardSize.Columns;
            }

            return new Move
            {
                Row = row,
                Column = col,
                Direction = dir,
                MoveIndex = moveIndex,
            };
        }

        /// <summary>
        /// Check if the move is valid for the given board size.
        /// This will be passed the return value from AbstractBoard.GetCurrentBoardSize().
        /// </summary>
        /// <param name="boardSize"></param>
        /// <returns></returns>
        public bool InRangeForBoard(BoardSize boardSize)
        {
            var (otherRow, otherCol) = OtherCell();
            // Get the maximum row and column this move would affect.
            var maxMoveRow = Mathf.Max(Row, otherRow);
            var maxMoveCol = Mathf.Max(Column, otherCol);
            return maxMoveRow < boardSize.Rows && maxMoveCol < boardSize.Columns;
        }

        /// <summary>
        /// Get the other row and column that correspond to this move.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public (int Row, int Column) OtherCell()
        {
            switch (Direction)
            {
                case Direction.Up:
                    return (Row + 1, Column);
                case Direction.Down:
                    return (Row - 1, Column);
                case Direction.Left:
                    return (Row, Column - 1);
                case Direction.Right:
                    return (Row, Column + 1);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        /// Get the opposite direction of this move.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Direction OtherDirection()
        {
            switch (Direction)
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
        /// Return the number of potential moves for a board of the given size.
        /// This is equivalent to the number of internal edges in the board.
        /// </summary>
        /// <param name="maxBoardSize"></param>
        /// <returns></returns>
        public static int NumPotentialMoves(BoardSize maxBoardSize)
        {
            return maxBoardSize.Rows * (maxBoardSize.Columns - 1) + (maxBoardSize.Rows - 1) * (maxBoardSize.Columns);
        }
    }
}
