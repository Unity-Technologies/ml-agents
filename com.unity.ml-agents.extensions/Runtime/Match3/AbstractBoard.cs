using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
{
    public abstract class AbstractBoard
    {
        public readonly int Rows;
        public readonly int Columns;
        public readonly int NumCellTypes;

        public AbstractBoard(int rows, int cols, int numCellTypes)
        {
            Rows = rows;
            Columns = cols;
            NumCellTypes = numCellTypes;
        }

        public abstract bool MakeMove(Move m);
        public abstract bool IsMoveValid(Move m);
        // TODO handle "special" cell types?
        public abstract int GetCellType(int row, int col);

        /// <summary>
        /// Returns a random valid move index, or -1 if none are available.
        /// </summary>
        /// <param name="rand"></param>
        /// <returns></returns>
        public int GetRandomValidMoveIndex(System.Random rand)
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
