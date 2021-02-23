using NUnit.Framework;
using Unity.MLAgents.Extensions.Match3;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Tests.Match3
{
    internal class SimpleBoard : AbstractBoard
    {
        public int LastMoveIndex;
        public bool MovesAreValid = true;

        public bool CallbackCalled;

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

    }
}
