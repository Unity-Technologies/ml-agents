using System;
using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgents.Extensions.Tests.Match3
{
    public class MoveTests
    {
        [Test]
        public void TestMoveEquivalence()
        {
            var moveUp = Move.FromPositionAndDirection(1, 1, Direction.Up, 10, 10);
            var moveDown = Move.FromPositionAndDirection(2, 1, Direction.Down, 10, 10);
            Assert.AreEqual(moveUp.MoveIndex, moveDown.MoveIndex);

            var moveRight = Move.FromPositionAndDirection(1, 1, Direction.Right, 10, 10);
            var moveLeft = Move.FromPositionAndDirection(1, 2, Direction.Left, 10, 10);
            Assert.AreEqual(moveRight.MoveIndex, moveLeft.MoveIndex);
        }

        [Test]
        public void TestAdvance()
        {
            var maxRows = 8;
            var maxCols = 13;
            // make sure using Advance agrees with FromMoveIndex.
            var advanceMove = Move.FromMoveIndex(0, maxRows, maxCols);
            for (var moveIndex = 0; moveIndex < Move.NumPotentialMoves(maxRows, maxCols); moveIndex++)
            {
                var moveFromIndex = Move.FromMoveIndex(moveIndex, maxRows, maxCols);
                Assert.AreEqual(advanceMove.MoveIndex, moveFromIndex.MoveIndex);
                Assert.AreEqual(advanceMove.Row, moveFromIndex.Row);
                Assert.AreEqual(advanceMove.Column, moveFromIndex.Column);
                Assert.AreEqual(advanceMove.Direction, moveFromIndex.Direction);

                advanceMove.Advance(maxRows, maxCols);
            }
        }
    }
}
