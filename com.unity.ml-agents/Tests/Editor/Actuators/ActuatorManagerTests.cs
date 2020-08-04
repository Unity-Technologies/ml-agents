using System;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Assert = UnityEngine.Assertions.Assert;

namespace Unity.MLAgents.Tests.Actuators
{
    internal class TestActuator : IActuator
    {
        public ActionBuffers LastActionBuffer;
        //public int branch;
        //public int[] MaskIndexes;
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
            // if (MaskIndexes != null)
            // {
            //     actionMask.WriteMask(branch, MaskIndexes);
            // }
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
            var actuator1 = new TestActuator(ActionSpaceDef.MakeDiscrete(new []{1 ,2, 3, 4}), "actuator1");
            var actuator2 = new TestActuator(ActionSpaceDef.MakeDiscrete(new [] {1, 1, 1}), "actuator2");
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
    }
}
