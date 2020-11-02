using System;

namespace Unity.MLAgents.Extensions.Match3
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
        /// <param name="maxRows"></param>
        /// <param name="maxCols"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static Move FromMoveIndex(int moveIndex, int maxRows, int maxCols)
        {
            if (moveIndex < 0 || moveIndex >= NumPotentialMoves(maxRows, maxCols))
            {
                throw new ArgumentOutOfRangeException("Invalid move index.");
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
        /// <param name="maxRows"></param>
        /// <param name="maxCols"></param>
        public void Next(int maxRows, int maxCols)
        {
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
        /// Construct a Move from the row, column, and direction.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="dir"></param>
        /// <param name="maxRows"></param>
        /// <param name="maxCols"></param>
        /// <returns></returns>
        public static Move FromPositionAndDirection(int row, int col, Direction dir, int maxRows, int maxCols)
        {

            // Check for out-of-bounds
            if (row < 0 || row >= maxRows)
            {
                throw new IndexOutOfRangeException($"row was {row}, but must be between 0 and {maxRows - 1}.");
            }

            if (col < 0 || col >= maxCols)
            {
                throw new IndexOutOfRangeException($"col was {col}, but must be between 0 and {maxCols - 1}.");
            }

            // Check moves that would go out of bounds e.g. col == 0 and dir == Left
            if (
                row == 0 && dir == Direction.Down ||
                row == maxRows - 1 && dir == Direction.Up ||
                col == 0 && dir == Direction.Left ||
                col == maxCols - 1 && dir == Direction.Right
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
                moveIndex = col + row * (maxCols - 1);
            }
            else
            {
                var offset = (maxCols - 1) * maxRows;
                moveIndex = offset + col + row * maxCols;
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
        /// <param name="maxRows"></param>
        /// <param name="maxCols"></param>
        /// <returns></returns>
        public static int NumPotentialMoves(int maxRows, int maxCols)
        {
            return maxRows * (maxCols - 1) + (maxRows - 1) * (maxCols);
        }
    }
}
