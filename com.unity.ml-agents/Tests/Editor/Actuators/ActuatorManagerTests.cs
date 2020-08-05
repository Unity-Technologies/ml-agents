using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.TestTools;
using Assert = UnityEngine.Assertions.Assert;

namespace Unity.MLAgents.Tests.Actuators
{
    internal class TestActuator : IActuator
    {
        public ActionBuffers LastActionBuffer;
        public int[][] Masks;
        public TestActuator(ActionSpaceDef actuatorSpace, string name)
        {
            ActionSpaceDef = actuatorSpace;
            TotalNumberOfActions = actuatorSpace.NumContinuousActions +
                actuatorSpace.NumDiscreteActions;
            Name = name;
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            LastActionBuffer = actionBuffers;
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            for (var i = 0; i < Masks.Length; i++)
            {
                actionMask.WriteMask(i, Masks[i]);
            }
        }

        public int TotalNumberOfActions { get; }
        public ActionSpaceDef ActionSpaceDef { get; }

        public string Name { get; }

        public void ResetData()
        {
        }
    }

    [TestFixture]
    public class ActuatorManagerTests
    {
        [Test]
        public void TestEnsureBufferSizeContinuous()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeContinuous(10), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(2), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var actuator1ActionSpaceDef = actuator1.ActionSpaceDef;
            var actuator2ActionSpaceDef = actuator2.ActionSpaceDef;
            manager.EnsureActionBufferSize(new[] { actuator1, actuator2 },
                actuator1ActionSpaceDef.NumContinuousActions + actuator2ActionSpaceDef.NumContinuousActions,
                actuator1ActionSpaceDef.SumOfDiscreteBranchSizes + actuator2ActionSpaceDef.SumOfDiscreteBranchSizes,
                actuator1ActionSpaceDef.NumDiscreteActions + actuator2ActionSpaceDef.NumDiscreteActions);

            manager.UpdateActions(new[]
                { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f }, Array.Empty<int>());

            Assert.IsTrue(12 == manager.NumContinuousActions);
            Assert.IsTrue(0 == manager.NumDiscreteBranches);
            Assert.IsTrue(0 == manager.SumOfDiscreteBranchSizes);
            Assert.IsTrue(12 == manager.StoredContinuousActions.Length);
            Assert.IsTrue(0 == manager.StoredDiscreteActions.Length);
        }

        [Test]
        public void TestEnsureBufferDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1 , 2, 3, 4}), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 1, 1}), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var actuator1ActionSpaceDef = actuator1.ActionSpaceDef;
            var actuator2ActionSpaceDef = actuator2.ActionSpaceDef;
            manager.EnsureActionBufferSize(new[] { actuator1, actuator2 },
                actuator1ActionSpaceDef.NumContinuousActions + actuator2ActionSpaceDef.NumContinuousActions,
                actuator1ActionSpaceDef.SumOfDiscreteBranchSizes + actuator2ActionSpaceDef.SumOfDiscreteBranchSizes,
                actuator1ActionSpaceDef.NumDiscreteActions + actuator2ActionSpaceDef.NumDiscreteActions);

            manager.UpdateActions(Array.Empty<float>(),
                new[] { 0, 1, 2, 3, 4, 5, 6});

            Assert.IsTrue(0 == manager.NumContinuousActions);
            Assert.IsTrue(7 == manager.NumDiscreteBranches);
            Assert.IsTrue(13 == manager.SumOfDiscreteBranchSizes);
            Assert.IsTrue(0 == manager.StoredContinuousActions.Length);
            Assert.IsTrue(7 == manager.StoredDiscreteActions.Length);
        }

        [Test]
        public void TestFailOnMixedActionSpace()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1 , 2, 3, 4}), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            manager.EnsureActionBufferSize(new[] { actuator1, actuator2 }, 3, 10, 4);
            LogAssert.Expect(LogType.Assert, "Actuators on the same Agent must have the same action SpaceType.");
        }

        [Test]
        public void TestFailOnSameActuatorName()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator1");
            manager.Add(actuator1);
            manager.Add(actuator2);
            manager.EnsureActionBufferSize(new[] { actuator1, actuator2 }, 3, 10, 4);
            LogAssert.Expect(LogType.Assert, "Actuator names must be unique.");
        }

        [Test]
        public void TestExecuteActionsDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1 , 2, 3, 4}), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 1, 1}), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            var discreteActionBuffer = new[] { 0, 1, 2, 3, 4, 5, 6};
            manager.UpdateActions(Array.Empty<float>(),
                discreteActionBuffer);

            manager.ExecuteActions();
            var actuator1Actions = actuator1.LastActionBuffer.DiscreteActions;
            var actuator2Actions = actuator2.LastActionBuffer.DiscreteActions;
            TestSegmentEquality(actuator1Actions, discreteActionBuffer); TestSegmentEquality(actuator2Actions, discreteActionBuffer);
        }

        [Test]
        public void TestExecuteActionsContinuous()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f};
            manager.UpdateActions(continuousActionBuffer,
                Array.Empty<int>());

            manager.ExecuteActions();
            var actuator1Actions = actuator1.LastActionBuffer.ContinuousActions;
            var actuator2Actions = actuator2.LastActionBuffer.ContinuousActions;
            TestSegmentEquality(actuator1Actions, continuousActionBuffer);
            TestSegmentEquality(actuator2Actions, continuousActionBuffer);
        }

        static void TestSegmentEquality<T>(ActionSegment<T> actionSegment, T[] actionBuffer)
            where T : struct
        {
            Assert.IsFalse(actionSegment.Length == 0);
            for (var i = 0; i < actionSegment.Length; i++)
            {
                var action = actionSegment[i];
                Assert.AreEqual(action, actionBuffer[actionSegment.Offset + i]);
            }
        }

        [Test]
        public void TestUpdateActionsContinuous()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f};
            manager.UpdateActions(continuousActionBuffer,
                Array.Empty<int>());

            Assert.IsTrue(manager.StoredContinuousActions.SequenceEqual(continuousActionBuffer));
        }

        [Test]
        public void TestUpdateActionsDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var discreteActionBuffer = new[] { 0, 1, 2, 3, 4, 5};
            manager.UpdateActions(Array.Empty<float>(),
                discreteActionBuffer);

            Debug.Log(manager.StoredDiscreteActions);
            Debug.Log(discreteActionBuffer);
            Assert.IsTrue(manager.StoredDiscreteActions.SequenceEqual(discreteActionBuffer));
        }

        [Test]
        public void TestRemove()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "actuator2");

            manager.Add(actuator1);
            manager.Add(actuator2);
            Assert.IsTrue(manager.NumDiscreteBranches == 6);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 12);

            manager.Remove(actuator2);

            Assert.IsTrue(manager.NumDiscreteBranches == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);

            manager.Remove(null);

            Assert.IsTrue(manager.NumDiscreteBranches == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);

            manager.RemoveAt(0);
            Assert.IsTrue(manager.NumDiscreteBranches == 0);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 0);
        }

        [Test]
        public void TestClear()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            Assert.IsTrue(manager.NumDiscreteBranches == 6);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 12);

            manager.Clear();

            Assert.IsTrue(manager.NumDiscreteBranches == 0);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 0);
        }

        [Test]
        public void TestIndexSet()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] { 1, 2, 3, 4}),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "actuator2");
            manager.Add(actuator1);
            Assert.IsTrue(manager.NumDiscreteBranches == 4);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 10);
            manager[0] = actuator2;
            Assert.IsTrue(manager.NumDiscreteBranches == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);
        }

        [Test]
        public void TestInsert()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] { 1, 2, 3, 4}),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "actuator2");
            manager.Add(actuator1);
            Assert.IsTrue(manager.NumDiscreteBranches == 4);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 10);
            manager.Insert(0, actuator2);
            Assert.IsTrue(manager.NumDiscreteBranches == 7);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 16);
            Assert.IsTrue(manager.IndexOf(actuator2) == 0);
        }

        [Test]
        public void TestResetData()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpaceDef.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f};
            manager.UpdateActions(continuousActionBuffer,
                Array.Empty<int>());

            Assert.IsTrue(manager.StoredContinuousActions.SequenceEqual(continuousActionBuffer));
            Assert.IsTrue(manager.NumContinuousActions == 6);
            manager.ResetData();

            Assert.IsTrue(manager.StoredContinuousActions.SequenceEqual(new[] { 0f, 0f, 0f, 0f, 0f, 0f}));
        }

        [Test]
        public void TestWriteDiscreteActionMask()
        {
            var manager = new ActuatorManager(2);
            var va1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {1, 2, 3}), "name");
            var va2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new[] {3, 2, 1}), "name1");
            manager.Add(va1);
            manager.Add(va2);

            var groundTruthMask = new[]
            {
                false,
                true, false,
                false, true, true,
                true, false, true,
                false, true,
                false
            };

            va1.Masks = new[]
            {
                Array.Empty<int>(),
                new[] { 0 },
                new[] { 1, 2 }
            };

            va2.Masks = new[]
            {
                new[] {0, 2},
                new[] {1},
                Array.Empty<int>()
            };
            manager.WriteActionMask();
            Assert.IsTrue(groundTruthMask.SequenceEqual(manager.DiscreteActionMask.GetMask()));
        }
    }
}
