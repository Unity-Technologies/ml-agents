using Unity.MLAgents.Integrations.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3ExampleActuator : Match3Actuator
    {
        private Match3Board m_Board;
        Match3Board Board => m_Board;

        public Match3ExampleActuator(Match3Board board,
            bool forceHeuristic,
            string name,
            int seed
        )
            : base(board, forceHeuristic, seed, name)
        {
            m_Board = board;
        }


        protected override int EvalMovePoints(Move move)
        {
            var pointsByType = new[] { Board.BasicCellPoints, Board.SpecialCell1Points, Board.SpecialCell2Points };
            // Counts the expected points for making the move.
            var moveVal = Board.GetCellType(move.Row, move.Column);
            var moveSpecial = Board.GetSpecialType(move.Row, move.Column);
            var (otherRow, otherCol) = move.OtherCell();
            var oppositeVal = Board.GetCellType(otherRow, otherCol);
            var oppositeSpecial = Board.GetSpecialType(otherRow, otherCol);


            int movePoints = EvalHalfMove(
                otherRow, otherCol, moveVal, moveSpecial, move.Direction, pointsByType
            );
            int otherPoints = EvalHalfMove(
                move.Row, move.Column, oppositeVal, oppositeSpecial, move.OtherDirection(), pointsByType
            );
            return movePoints + otherPoints;
        }

        int EvalHalfMove(int newRow, int newCol, int newValue, int newSpecial, Direction incomingDirection, int[] pointsByType)
        {
            // This is a essentially a duplicate of AbstractBoard.CheckHalfMove but also counts the points for the move.
            var currentBoardSize = Board.GetCurrentBoardSize();
            int matchedLeft = 0, matchedRight = 0, matchedUp = 0, matchedDown = 0;
            int scoreLeft = 0, scoreRight = 0, scoreUp = 0, scoreDown = 0;

            if (incomingDirection != Direction.Right)
            {
                for (var c = newCol - 1; c >= 0; c--)
                {
                    if (Board.GetCellType(newRow, c) == newValue)
                    {
                        matchedLeft++;
                        scoreLeft += pointsByType[Board.GetSpecialType(newRow, c)];
                    }
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Left)
            {
                for (var c = newCol + 1; c < currentBoardSize.Columns; c++)
                {
                    if (Board.GetCellType(newRow, c) == newValue)
                    {
                        matchedRight++;
                        scoreRight += pointsByType[Board.GetSpecialType(newRow, c)];
                    }
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Down)
            {
                for (var r = newRow + 1; r < currentBoardSize.Rows; r++)
                {
                    if (Board.GetCellType(r, newCol) == newValue)
                    {
                        matchedUp++;
                        scoreUp += pointsByType[Board.GetSpecialType(r, newCol)];
                    }
                    else
                        break;
                }
            }

            if (incomingDirection != Direction.Up)
            {
                for (var r = newRow - 1; r >= 0; r--)
                {
                    if (Board.GetCellType(r, newCol) == newValue)
                    {
                        matchedDown++;
                        scoreDown += pointsByType[Board.GetSpecialType(r, newCol)];
                    }
                    else
                        break;
                }
            }

            if ((matchedUp + matchedDown >= 2) || (matchedLeft + matchedRight >= 2))
            {
                // It's a match. Start from counting the piece being moved
                var totalScore = pointsByType[newSpecial];
                if (matchedUp + matchedDown >= 2)
                {
                    totalScore += scoreUp + scoreDown;
                }

                if (matchedLeft + matchedRight >= 2)
                {
                    totalScore += scoreLeft + scoreRight;
                }
                return totalScore;
            }

            return 0;
        }
    }

}
