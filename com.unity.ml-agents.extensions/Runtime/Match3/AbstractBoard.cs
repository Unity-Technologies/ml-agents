using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
{
    public abstract class AbstractBoard : MonoBehaviour
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
        /// Returns the "color" of the piece at the given row and column.
        /// This should be between 0 and NumCellTypes-1 (inclusive).
        /// The actual order of the values doesn't matter.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public abstract int GetCellType(int row, int col);

        /// <summary>
        /// Returns the special type of the piece at the given row and column.
        /// This should be between 0 and NumSpecialTypes (inclusive).
        /// The actual order of the values doesn't matter.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public abstract int GetSpecialType(int row, int col);

        /// <summary>
        /// Check whether the particular Move is valid for the game.
        /// The actual results will depend on the rules of the game, but we provide SimpleIsMoveValid()
        /// that handles basic match3 rules with no special or immovable pieces.
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public abstract bool IsMoveValid(Move m);

        /// <summary>
        /// Instruct the game to make the given move. Returns true if the move was made.
        /// Note that during training, a move that was marked as invalid may occasionally still be
        /// requested. If this happens, it is safe to do nothing and request another move.
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public abstract bool MakeMove(Move m);

        /// <summary>
        /// Return the total number of moves possible for the board.
        /// </summary>
        /// <returns></returns>
        public int NumMoves()
        {
            return Move.NumPotentialMoves(Rows, Columns);
        }

        /// <summary>
        /// An optional callback for when the all moves are invalid. Ideally, the game state should
        /// be changed before this happens, but this is a way to get notified if not.
        /// </summary>
        public Action OnNoValidMovesAction;

        /// <summary>
        /// Iterate through all Moves on the board.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Move> AllMoves()
        {
            var currentMove = Move.FromMoveIndex(0, Rows, Columns);
            for (var i = 0; i < NumMoves(); i++)
            {
                yield return currentMove;
                currentMove.Next(Rows, Columns);
            }
        }

        /// <summary>
        /// Iterate through all valid Moves on the board.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Move> ValidMoves()
        {
            var currentMove = Move.FromMoveIndex(0, Rows, Columns);
            for (var i = 0; i < NumMoves(); i++)
            {
                if (IsMoveValid(currentMove))
                {
                    yield return currentMove;
                }
                currentMove.Next(Rows, Columns);
            }
        }

        /// <summary>
        /// Iterate through all invalid Moves on the board.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Move> InvalidMoves()
        {
            var currentMove = Move.FromMoveIndex(0, Rows, Columns);
            for (var i = 0; i < NumMoves(); i++)
            {
                if (!IsMoveValid(currentMove))
                {
                    yield return currentMove;
                }
                currentMove.Next(Rows, Columns);
            }
        }

        /// <summary>
        /// Returns true if swapped the cells specified by the move would result in
        /// 3 or more cells of the same type in a row. This assumes that all pieces are allowed
        /// to be moved; to add extra logic, incorporate it into you IsMoveValid() method.
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
        /// Since these checks are similar for each cell, we consider the Move as two "half moves".
        /// </summary>
        /// <param name="newRow"></param>
        /// <param name="newCol"></param>
        /// <param name="newValue"></param>
        /// <param name="incomingDirection"></param>
        /// <returns></returns>
        bool CheckHalfMove(int newRow, int newCol, int newValue, Direction incomingDirection)
        {
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
                for (var c = newCol + 1; c < Columns; c++)
                {
                    if (GetCellType(newRow, c) == newValue)
                        matchedRight++;
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Down)
            {
                for (var r = newRow + 1; r < Rows; r++)
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

    }
}
