using System;
using NUnit.Framework;
using Unity.MLAgents.Integrations.Match3;

namespace Unity.MLAgents.Tests.Integrations.Match3
{
    public class MoveTests
    {
        [Test]
        public void TestMoveEquivalence()
        {
            var board10x10 = new BoardSize { Rows = 10, Columns = 10 };
            var moveUp = Move.FromPositionAndDirection(1, 1, Direction.Up, board10x10);
            var moveDown = Move.FromPositionAndDirection(2, 1, Direction.Down, board10x10);
            Assert.AreEqual(moveUp.MoveIndex, moveDown.MoveIndex);

            var moveRight = Move.FromPositionAndDirection(1, 1, Direction.Right, board10x10);
            var moveLeft = Move.FromPositionAndDirection(1, 2, Direction.Left, board10x10);
            Assert.AreEqual(moveRight.MoveIndex, moveLeft.MoveIndex);
        }

        [Test]
        public void TestNext()
        {
            var maxRows = 8;
            var maxCols = 13;
            var boardSize = new BoardSize
            {
                Rows = maxRows,
                Columns = maxCols
            };
            // make sure using Next agrees with FromMoveIndex.
            var advanceMove = Move.FromMoveIndex(0, boardSize);
            for (var moveIndex = 0; moveIndex < Move.NumPotentialMoves(boardSize); moveIndex++)
            {
                var moveFromIndex = Move.FromMoveIndex(moveIndex, boardSize);
                Assert.AreEqual(advanceMove.MoveIndex, moveFromIndex.MoveIndex);
                Assert.AreEqual(advanceMove.Row, moveFromIndex.Row);
                Assert.AreEqual(advanceMove.Column, moveFromIndex.Column);
                Assert.AreEqual(advanceMove.Direction, moveFromIndex.Direction);

                advanceMove.Next(boardSize);
            }
        }

        // These are off the board
        [TestCase(-1, 5, Direction.Up)]
        [TestCase(10, 5, Direction.Up)]
        [TestCase(5, -1, Direction.Up)]
        [TestCase(5, 10, Direction.Up)]
        // These are on the board but would move off
        [TestCase(0, 5, Direction.Down)]
        [TestCase(9, 5, Direction.Up)]
        [TestCase(5, 0, Direction.Left)]
        [TestCase(5, 9, Direction.Right)]
        public void TestInvalidMove(int row, int col, Direction dir)
        {
            var board10x10 = new BoardSize { Rows = 10, Columns = 10 };
            Assert.Throws<IndexOutOfRangeException>(() =>
            {
                Move.FromPositionAndDirection(row, col, dir, board10x10);
            });

        }
    }
}
