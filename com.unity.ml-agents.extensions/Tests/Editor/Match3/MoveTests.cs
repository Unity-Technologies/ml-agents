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
            Assert.AreEqual(moveUp.InternalEdgeIndex, moveDown.InternalEdgeIndex);

            var moveRight = Move.FromPositionAndDirection(1, 1, Direction.Right, 10, 10);
            var moveLeft = Move.FromPositionAndDirection(1, 2, Direction.Left, 10, 10);
            Assert.AreEqual(moveRight.InternalEdgeIndex, moveLeft.InternalEdgeIndex);

        }
    }
}
