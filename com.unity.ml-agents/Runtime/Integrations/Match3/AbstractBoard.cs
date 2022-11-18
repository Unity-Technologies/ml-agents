using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Representation of the AbstractBoard dimensions, and number of cell and special types.
    /// </summary>
    public struct BoardSize
    {
        /// <summary>
        /// Number of rows on the board
        /// </summary>
        public int Rows;

        /// <summary>
        /// Number of columns on the board
        /// </summary>
        public int Columns;

        /// <summary>
        /// Maximum number of different types of cells (colors, pieces, etc).
        /// </summary>
        public int NumCellTypes;

        /// <summary>
        /// Maximum number of special types. This can be zero, in which case
        /// all cells of the same type are assumed to be equivalent.
        /// </summary>
        public int NumSpecialTypes;

        /// <summary>
        /// Check that all fields of the left-hand BoardSize are less than or equal to the field of the right-hand BoardSize
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>True if all fields are less than or equal.</returns>
        public static bool operator <=(BoardSize lhs, BoardSize rhs)
        {
            return lhs.Rows <= rhs.Rows && lhs.Columns <= rhs.Columns && lhs.NumCellTypes <= rhs.NumCellTypes &&
                lhs.NumSpecialTypes <= rhs.NumSpecialTypes;
        }

        /// <summary>
        /// Check that all fields of the left-hand BoardSize are greater than or equal to the field of the right-hand BoardSize
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>True if all fields are greater than or equal.</returns>
        public static bool operator >=(BoardSize lhs, BoardSize rhs)
        {
            return lhs.Rows >= rhs.Rows && lhs.Columns >= rhs.Columns && lhs.NumCellTypes >= rhs.NumCellTypes &&
                lhs.NumSpecialTypes >= rhs.NumSpecialTypes;
        }

        /// <summary>
        /// Return a string representation of the BoardSize.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return
                $"Rows: {Rows}, Columns: {Columns}, NumCellTypes: {NumCellTypes}, NumSpecialTypes: {NumSpecialTypes}";
        }
    }

    /// <summary>
    /// An adapter between ML Agents and a Match-3 game.
    /// </summary>
    public abstract class AbstractBoard : MonoBehaviour
    {
        /// <summary>
        /// Return the maximum size of the board. This is used to determine the size of observations and actions,
        /// so the returned values must not change.
        /// </summary>
        /// <returns></returns>
        public abstract BoardSize GetMaxBoardSize();

        /// <summary>
        /// Return the current size of the board. The values must less than or equal to the values returned from
        /// <see cref="GetMaxBoardSize"/>.
        /// By default, this will return <see cref="GetMaxBoardSize"/>; if your board doesn't change size, you don't need to
        /// override it.
        /// </summary>
        /// <returns></returns>
        public virtual BoardSize GetCurrentBoardSize()
        {
            return GetMaxBoardSize();
        }

        /// <summary>
        /// Returns the "color" of the piece at the given row and column.
        /// This should be between 0 and BoardSize.NumCellTypes-1 (inclusive).
        /// The actual order of the values doesn't matter.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public abstract int GetCellType(int row, int col);

        /// <summary>
        /// Returns the special type of the piece at the given row and column.
        /// This should be between 0 and BoardSize.NumSpecialTypes (inclusive).
        /// The actual order of the values doesn't matter.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public abstract int GetSpecialType(int row, int col);

        /// <summary>
        /// Check whether the particular Move is valid for the game.
        /// The actual results will depend on the rules of the game, but we provide <see cref="SimpleIsMoveValid(Move)"/>
        /// that handles basic match3 rules with no special or immovable pieces.
        /// </summary>
        /// <remarks>
        /// Moves that would go outside of <see cref="GetCurrentBoardSize"/> are filtered out before they are
        /// passed to IsMoveValid().
        /// </remarks>
        /// <param name="m">The move to check.</param>
        /// <returns></returns>
        public abstract bool IsMoveValid(Move m);

        /// <summary>
        /// Instruct the game to make the given <see cref="Move"/>. Returns true if the move was made.
        /// Note that during training, a move that was marked as invalid may occasionally still be
        /// requested. If this happens, it is safe to do nothing and request another move.
        /// </summary>
        /// <param name="m">The move to carry out.</param>
        /// <returns></returns>
        public abstract bool MakeMove(Move m);

        /// <summary>
        /// Return the total number of moves possible for the board.
        /// </summary>
        /// <returns></returns>
        public int NumMoves()
        {
            return Move.NumPotentialMoves(GetMaxBoardSize());
        }

        /// <summary>
        /// An optional callback for when the all moves are invalid. Ideally, the game state should
        /// be changed before this happens, but this is a way to get notified if not.
        /// </summary>
        public Action OnNoValidMovesAction;

        /// <summary>
        /// Iterate through all moves on the board.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Move> AllMoves()
        {
            var maxBoardSize = GetMaxBoardSize();
            var currentBoardSize = GetCurrentBoardSize();

            var currentMove = Move.FromMoveIndex(0, maxBoardSize);
            for (var i = 0; i < NumMoves(); i++)
            {
                if (currentMove.InRangeForBoard(currentBoardSize))
                {
                    yield return currentMove;
                }
                currentMove.Next(maxBoardSize);
            }
        }

        /// <summary>
        /// Iterate through all valid moves on the board.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Move> ValidMoves()
        {
            var maxBoardSize = GetMaxBoardSize();
            var currentBoardSize = GetCurrentBoardSize();

            var currentMove = Move.FromMoveIndex(0, maxBoardSize);
            for (var i = 0; i < NumMoves(); i++)
            {
                if (currentMove.InRangeForBoard(currentBoardSize) && IsMoveValid(currentMove))
                {
                    yield return currentMove;
                }
                currentMove.Next(maxBoardSize);
            }
        }

        /// <summary>
        /// Returns true if swapping the cells specified by the move would result in
        /// 3 or more cells of the same type in a row. This assumes that all pieces are allowed
        /// to be moved; to add extra logic, incorporate it into your <see cref="IsMoveValid"/> method.
        /// </summary>
        /// <param name="move"></param>
        /// <returns></returns>
        public bool SimpleIsMoveValid(Move move)
        {
            using (TimerStack.Instance.Scoped("SimpleIsMoveValid"))
            {
                var moveVal = GetCellType(move.Row, move.Column);
                var (otherRow, otherCol) = move.OtherCell();
                var oppositeVal = GetCellType(otherRow, otherCol);

                // Simple check - if the values are the same, don't match
                // This might not be valid for all games
                {
                    if (moveVal == oppositeVal)
                    {
                        return false;
                    }
                }

                bool moveMatches = CheckHalfMove(otherRow, otherCol, moveVal, move.Direction);
                if (moveMatches)
                {
                    // early out
                    return true;
                }

                bool otherMatches = CheckHalfMove(move.Row, move.Column, oppositeVal, move.OtherDirection());
                return otherMatches;
            }
        }

        /// <summary>
        /// Check if one of the cells that is swapped during a move matches 3 or more.
        /// Since these checks are similar for each cell, we consider the move as two "half moves".
        /// </summary>
        /// <param name="newRow"></param>
        /// <param name="newCol"></param>
        /// <param name="newValue"></param>
        /// <param name="incomingDirection"></param>
        /// <returns></returns>
        bool CheckHalfMove(int newRow, int newCol, int newValue, Direction incomingDirection)
        {
            var currentBoardSize = GetCurrentBoardSize();
            int matchedLeft = 0, matchedRight = 0, matchedUp = 0, matchedDown = 0;

            if (incomingDirection != Direction.Right)
            {
                for (var c = newCol - 1; c >= 0; c--)
                {
                    if (GetCellType(newRow, c) == newValue)
                        matchedLeft++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Left)
            {
                for (var c = newCol + 1; c < currentBoardSize.Columns; c++)
                {
                    if (GetCellType(newRow, c) == newValue)
                        matchedRight++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Down)
            {
                for (var r = newRow + 1; r < currentBoardSize.Rows; r++)
                {
                    if (GetCellType(r, newCol) == newValue)
                        matchedUp++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Up)
            {
                for (var r = newRow - 1; r >= 0; r--)
                {
                    if (GetCellType(r, newCol) == newValue)
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

        /// <summary>
        /// Make sure that the current BoardSize isn't larger than the original value of <see cref="GetMaxBoardSize"/>.
        /// If it is, log a warning.
        /// </summary>
        /// <param name="originalMaxBoardSize"></param>
        [Conditional("DEBUG")]
        internal void CheckBoardSizes(BoardSize originalMaxBoardSize)
        {
            var currentBoardSize = GetCurrentBoardSize();
            if (!(currentBoardSize <= originalMaxBoardSize))
            {
                Debug.LogWarning(
                    "Current BoardSize is larger than maximum board size was on initialization. This may cause unexpected results.\n" +
                    $"Original GetMaxBoardSize() result: {originalMaxBoardSize}\n" +
                    $"GetCurrentBoardSize() result: {currentBoardSize}"
                );
            }
        }
    }
}
