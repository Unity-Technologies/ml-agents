using System;

namespace Unity.MLAgents.Extensions.Match3
{
    public enum Direction
    {
        Up, // +row direction
        Down, // -row direction
        Left, // -column direction
        Right, // +column direction
    }

    /// <summary>
    /// Struct that encapsulates a swap of adjacent cells.
    /// A Move can be constructed from either a starting row, column, and direction,
    /// or enumerated from 0 to NumPotentialMoves()-1
    /// </summary>
    public struct Move
    {
        /**
         * Moves are enumerated as the internal edges of the game grid.
         * Left/right moves come first. There are (maxCols - 1) * maxRows of these.
         * Up/down moves are next. There are (maxRows - 1) * maxCols of these.
         */
        public int InternalEdgeIndex;
        public int Row;
        public int Column;
        public Direction Direction;

        /// <summary>
        /// Construct a Move from its index and the board size.
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
                InternalEdgeIndex = moveIndex,
                Direction = dir,
                Row = row,
                Column = col
            };
        }

        public void Advance(int maxRows, int maxCols)
        {
            var switchoverIndex = (maxCols - 1) * maxRows;

            InternalEdgeIndex++;
            if (InternalEdgeIndex < switchoverIndex)
            {
                Column++;
                if (Column == maxCols - 1)
                {
                    Row++;
                    Column = 0;
                }
            }
            else if (InternalEdgeIndex == switchoverIndex)
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
            int edgeIndex;
            // Normalize - only consider Right and Up
            // TODO throw if e.g. col == 0 and dir == Left, etc.
            // TODO throw if row < 0 or row>=maxRows (and same for columns).
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
                Row = row,
                Column = col,
                Direction = dir,
                InternalEdgeIndex = edgeIndex,
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
