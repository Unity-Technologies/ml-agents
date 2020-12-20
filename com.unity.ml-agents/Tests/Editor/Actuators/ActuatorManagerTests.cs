using System;
using System.Linq;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.TestTools;
using Assert = UnityEngine.Assertions.Assert;

namespace Unity.MLAgents.Tests.Actuators
{
    [TestFixture]
    public class ActuatorManagerTests
    {
        [Test]
        public void TestEnsureBufferSizeContinuous()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeContinuous(10), "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(2), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var actuator1ActionSpaceDef = actuator1.ActionSpec;
            var actuator2ActionSpaceDef = actuator2.ActionSpec;
            manager.ReadyActuatorsForExecution(new[] { actuator1, actuator2 },
                actuator1ActionSpaceDef.NumContinuousActions + actuator2ActionSpaceDef.NumContinuousActions,
                actuator1ActionSpaceDef.SumOfDiscreteBranchSizes + actuator2ActionSpaceDef.SumOfDiscreteBranchSizes,
                actuator1ActionSpaceDef.NumDiscreteActions + actuator2ActionSpaceDef.NumDiscreteActions);

            manager.UpdateActions(new ActionBuffers(new[]
                { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f }, Array.Empty<int>()));

            Assert.IsTrue(12 == manager.NumContinuousActions);
            Assert.IsTrue(0 == manager.NumDiscreteActions);
            Assert.IsTrue(0 == manager.SumOfDiscreteBranchSizes);
            Assert.IsTrue(12 == manager.StoredActions.ContinuousActions.Length);
            Assert.IsTrue(0 == manager.StoredActions.DiscreteActions.Length);
        }

        [Test]
        public void TestEnsureBufferDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3, 4 }), "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 1, 1 }), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var actuator1ActionSpaceDef = actuator1.ActionSpec;
            var actuator2ActionSpaceDef = actuator2.ActionSpec;
            manager.ReadyActuatorsForExecution(new[] { actuator1, actuator2 },
                actuator1ActionSpaceDef.NumContinuousActions + actuator2ActionSpaceDef.NumContinuousActions,
                actuator1ActionSpaceDef.SumOfDiscreteBranchSizes + actuator2ActionSpaceDef.SumOfDiscreteBranchSizes,
                actuator1ActionSpaceDef.NumDiscreteActions + actuator2ActionSpaceDef.NumDiscreteActions);

            manager.UpdateActions(new ActionBuffers(Array.Empty<float>(),
                new[] { 0, 1, 2, 3, 4, 5, 6 }));

            Assert.IsTrue(0 == manager.NumContinuousActions);
            Assert.IsTrue(7 == manager.NumDiscreteActions);
            Assert.IsTrue(13 == manager.SumOfDiscreteBranchSizes);
            Assert.IsTrue(0 == manager.StoredActions.ContinuousActions.Length);
            Assert.IsTrue(7 == manager.StoredActions.DiscreteActions.Length);
        }

        [Test]
        public void TestAllowMixedActions()
        {
            // Make sure discrete + continuous actuators are allowed.
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3, 4 }), "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            manager.ReadyActuatorsForExecution(new[] { actuator1, actuator2 }, 3, 10, 4);
        }

        [Test]
        public void TestFailOnSameActuatorName()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator1");
            manager.Add(actuator1);
            manager.Add(actuator2);
            manager.ReadyActuatorsForExecution(new[] { actuator1, actuator2 }, 3, 10, 4);
            LogAssert.Expect(LogType.Assert, "Actuator names must be unique.");
        }

        [Test]
        public void TestExecuteActionsDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3, 4 }), "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 1, 1 }), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            var discreteActionBuffer = new[] { 0, 1, 2, 3, 4, 5, 6 };
            manager.UpdateActions(new ActionBuffers(Array.Empty<float>(),
                discreteActionBuffer));

            manager.ExecuteActions();
            var actuator1Actions = actuator1.LastActionBuffer.DiscreteActions;
            var actuator2Actions = actuator2.LastActionBuffer.DiscreteActions;
            TestSegmentEquality(actuator1Actions, discreteActionBuffer); TestSegmentEquality(actuator2Actions, discreteActionBuffer);
        }

        [Test]
        public void TestExecuteActionsContinuous()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f };
            manager.UpdateActions(new ActionBuffers(continuousActionBuffer,
                Array.Empty<int>()));

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
            var actuator1 = new TestActuator(ActionSpec.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f };
            manager.UpdateActions(new ActionBuffers(continuousActionBuffer,
                Array.Empty<int>()));

            Assert.IsTrue(manager.StoredActions.ContinuousActions.SequenceEqual(continuousActionBuffer));
        }

        [Test]
        public void TestUpdateActionsDiscrete()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var discreteActionBuffer = new[] { 0, 1, 2, 3, 4, 5 };
            manager.UpdateActions(new ActionBuffers(Array.Empty<float>(),
                discreteActionBuffer));

            Debug.Log(manager.StoredActions.DiscreteActions);
            Debug.Log(discreteActionBuffer);
            Assert.IsTrue(manager.StoredActions.DiscreteActions.SequenceEqual(discreteActionBuffer));
        }

        [Test]
        public void TestRemove()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "actuator2");

            manager.Add(actuator1);
            manager.Add(actuator2);
            Assert.IsTrue(manager.NumDiscreteActions == 6);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 12);

            manager.Remove(actuator2);

            Assert.IsTrue(manager.NumDiscreteActions == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);

            manager.Remove(null);

            Assert.IsTrue(manager.NumDiscreteActions == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);

            manager.RemoveAt(0);
            Assert.IsTrue(manager.NumDiscreteActions == 0);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 0);
        }

        [Test]
        public void TestClear()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);

            Assert.IsTrue(manager.NumDiscreteActions == 6);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 12);

            manager.Clear();

            Assert.IsTrue(manager.NumDiscreteActions == 0);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 0);
        }

        [Test]
        public void TestIndexSet()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3, 4 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "actuator2");
            manager.Add(actuator1);
            Assert.IsTrue(manager.NumDiscreteActions == 4);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 10);
            manager[0] = actuator2;
            Assert.IsTrue(manager.NumDiscreteActions == 3);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 6);
        }

        [Test]
        public void TestInsert()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3, 4 }),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "actuator2");
            manager.Add(actuator1);
            Assert.IsTrue(manager.NumDiscreteActions == 4);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 10);
            manager.Insert(0, actuator2);
            Assert.IsTrue(manager.NumDiscreteActions == 7);
            Assert.IsTrue(manager.SumOfDiscreteBranchSizes == 16);
            Assert.IsTrue(manager.IndexOf(actuator2) == 0);
        }

        [Test]
        public void TestResetData()
        {
            var manager = new ActuatorManager();
            var actuator1 = new TestActuator(ActionSpec.MakeContinuous(3),
                "actuator1");
            var actuator2 = new TestActuator(ActionSpec.MakeContinuous(3), "actuator2");
            manager.Add(actuator1);
            manager.Add(actuator2);
            var continuousActionBuffer = new[] { 0f, 1f, 2f, 3f, 4f, 5f };
            manager.UpdateActions(new ActionBuffers(continuousActionBuffer,
                Array.Empty<int>()));

            Assert.IsTrue(manager.StoredActions.ContinuousActions.SequenceEqual(continuousActionBuffer));
            Assert.IsTrue(manager.NumContinuousActions == 6);
            manager.ResetData();

            Assert.IsTrue(manager.StoredActions.ContinuousActions.SequenceEqual(new[] { 0f, 0f, 0f, 0f, 0f, 0f }));
        }

        [Test]
        public void TestWriteDiscreteActionMask()
        {
            var manager = new ActuatorManager(2);
            var va1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 1, 2, 3 }), "name");
            var va2 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 3, 2, 1 }), "name1");
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
