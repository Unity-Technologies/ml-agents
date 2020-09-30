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
        public int InternalEdgeIndex;
        public int Row;
        public int Column;
        public Direction Direction;

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
                InternalEdgeIndex = edgeIndex,
                Direction = dir,
                Row = row,
                Column = col
            };
        }

        public static Move FromPositionAndDirection(int row, int col, Direction dir, int maxRows, int maxCols)
        {
            int edgeIndex;
            // Normalize - only consider Right and Up
            // TODO throw if e.g. col == 0 and dir == Left
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
}
