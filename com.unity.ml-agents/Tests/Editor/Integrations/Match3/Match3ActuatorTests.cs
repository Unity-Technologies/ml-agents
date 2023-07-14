using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Integrations.Match3;
using UnityEngine;

namespace Unity.MLAgents.Tests.Integrations.Match3
{
    internal class SimpleBoard : AbstractBoard
    {
        public int Rows;
        public int Columns;
        public int NumCellTypes;
        public int NumSpecialTypes;

        public int LastMoveIndex;
        public bool MovesAreValid = true;

        public bool CallbackCalled;

        public override BoardSize GetMaxBoardSize()
        {
            return new BoardSize
            {
                Rows = Rows,
                Columns = Columns,
                NumCellTypes = NumCellTypes,
                NumSpecialTypes = NumSpecialTypes
            };
        }

        public override int GetCellType(int row, int col)
        {
            return 0;
        }

        public override int GetSpecialType(int row, int col)
        {
            return 0;
        }

        public override bool IsMoveValid(Move m)
        {
            return MovesAreValid;
        }

        public override bool MakeMove(Move m)
        {
            LastMoveIndex = m.MoveIndex;
            return MovesAreValid;
        }

        public void Callback()
        {
            CallbackCalled = true;
        }
    }

    public class Match3ActuatorTests
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [TestCase(true)]
        [TestCase(false)]
        public void TestValidMoves(bool movesAreValid)
        {
            // Check that a board with no valid moves doesn't raise an exception.
            var gameObj = new GameObject();
            var board = gameObj.AddComponent<SimpleBoard>();
            var agent = gameObj.AddComponent<Agent>();
            gameObj.AddComponent<Match3ActuatorComponent>();

            board.Rows = 5;
            board.Columns = 5;
            board.NumCellTypes = 5;
            board.NumSpecialTypes = 0;

            board.MovesAreValid = movesAreValid;
            board.OnNoValidMovesAction = board.Callback;
            board.LastMoveIndex = -1;

            agent.LazyInitialize();
            agent.RequestDecision();
            Academy.Instance.EnvironmentStep();

            if (movesAreValid)
            {
                Assert.IsFalse(board.CallbackCalled);
            }
            else
            {
                Assert.IsTrue(board.CallbackCalled);
            }
            Assert.AreNotEqual(-1, board.LastMoveIndex);
        }

        [Test]
        public void TestActionSpec()
        {
            var gameObj = new GameObject();
            var board = gameObj.AddComponent<SimpleBoard>();
            var actuator = gameObj.AddComponent<Match3ActuatorComponent>();

            board.Rows = 5;
            board.Columns = 5;
            board.NumCellTypes = 5;
            board.NumSpecialTypes = 0;

            var actionSpec = actuator.ActionSpec;
            Assert.AreEqual(1, actionSpec.NumDiscreteActions);
            Assert.AreEqual(board.NumMoves(), actionSpec.BranchSizes[0]);
        }

        [Test]
        public void TestActionSpecNullBoard()
        {
            var gameObj = new GameObject();
            var actuator = gameObj.AddComponent<Match3ActuatorComponent>();

            var actionSpec = actuator.ActionSpec;
            Assert.AreEqual(0, actionSpec.NumDiscreteActions);
            Assert.AreEqual(0, actionSpec.NumContinuousActions);
        }

        public class HashSetActionMask : IDiscreteActionMask
        {
            public HashSet<int>[] HashSets;
            public HashSetActionMask(ActionSpec spec)
            {
                HashSets = new HashSet<int>[spec.NumDiscreteActions];
                for (var i = 0; i < spec.NumDiscreteActions; i++)
                {
                    HashSets[i] = new HashSet<int>();
                }
            }

            public void SetActionEnabled(int branch, int actionIndex, bool isEnabled)
            {
                var hashSet = HashSets[branch];
                if (isEnabled)
                {
                    hashSet.Remove(actionIndex);
                }
                else
                {
                    hashSet.Add(actionIndex);
                }
            }
        }

        [TestCase(true, TestName = "Full Board")]
        [TestCase(false, TestName = "Small Board")]
        public void TestMasking(bool fullBoard)
        {
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();

            var boardString =
@"0105
                  1024
                  0203
                  2022";
            board.SetBoard(boardString);
            var boardSize = board.GetMaxBoardSize();
            if (!fullBoard)
            {
                board.CurrentRows -= 1;
            }

            var validMoves = AbstractBoardTests.GetValidMoves4x4(fullBoard, boardSize);

            var actuatorComponent = gameObj.AddComponent<Match3ActuatorComponent>();
            var actuator = actuatorComponent.CreateActuators()[0];

            var masks = new HashSetActionMask(actuator.ActionSpec);
            actuator.WriteDiscreteActionMask(masks);

            // Run through all moves and make sure those are the only valid ones
            HashSet<int> validIndices = new HashSet<int>();
            foreach (var m in validMoves)
            {
                validIndices.Add(m.MoveIndex);
            }

            // Valid moves and masked moves should be disjoint
            Assert.IsFalse(validIndices.Overlaps(masks.HashSets[0]));
            // And they should add up to all the potential moves
            Assert.AreEqual(validIndices.Count + masks.HashSets[0].Count, board.NumMoves());
        }

        [Test]
        public void TestNoBoardReturnsEmptyActuators()
        {
            var gameObj = new GameObject("board");
            var actuatorComponent = gameObj.AddComponent<Match3ActuatorComponent>();
            var actuators = actuatorComponent.CreateActuators();
            Assert.AreEqual(0, actuators.Length);
        }
    }
}
