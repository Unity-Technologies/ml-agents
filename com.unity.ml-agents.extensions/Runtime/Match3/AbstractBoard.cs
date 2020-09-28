using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
{
    public abstract class AbstractBoard : MonoBehaviour
    {
        public int Rows;
        public int Columns;
        public int NumCellTypes;

        // public AbstractBoard(int rows, int cols, int numCellTypes)
        // {
        //     Rows = rows;
        //     Columns = cols;
        //     NumCellTypes = numCellTypes;
        // }

        public abstract bool MakeMove(Move m);
        public abstract bool IsMoveValid(Move m);
        // TODO handle "special" cell types?
        public abstract int GetCellType(int row, int col);

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
        /// Check if the "half" of a move is matches 3 or more.
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

        /// <summary>
        /// Returns a random valid move index, or -1 if none are available.
        /// </summary>
        /// <param name="rand"></param>
        /// <returns></returns>
        public int GetRandomValidMoveIndex(System.Random rand)
        {
            using (TimerStack.Instance.Scoped("GetRandomValidMove"))
            {
                int numMoves = Move.NumEdgeIndices(Rows, Columns);
                var validMoves = new bool[numMoves];

                int numValidMoves = 0;
                for (var index = 0; index < Move.NumEdgeIndices(Rows, Columns); index++)
                {
                    var move = Move.FromEdgeIndex(index, Rows, Columns);
                    if (IsMoveValid(move))
                    {
                        validMoves[index] = true;
                        numValidMoves++;
                    }
                }

                // TODO reservoir sample? More random calls, but one pass through the indices.
                if (numValidMoves == 0)
                {
                    Debug.Log("No valid moves");
                    return -1;
                }

                // We'll make the n'th valid move where n in [0, numValidMoves)
                var target = rand.Next(numValidMoves);
                var numSkipped = 0;

                for (var i = 0; i < validMoves.Length; i++)
                {
                    var valid = validMoves[i];
                    if (valid)
                    {
                        if (numSkipped == target)
                        {
                            return i;
                        }

                        numSkipped++;
                    }
                }

                // Should never reach here
                return -1;
            }
        }
    }
}
