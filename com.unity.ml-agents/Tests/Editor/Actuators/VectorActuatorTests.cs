using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Assert = UnityEngine.Assertions.Assert;

namespace Unity.MLAgents.Tests.Actuators
{
    [TestFixture]
    public class VectorActuatorTests
    {
        class TestActionReceiver : IActionReceiver
        {
            public ActionBuffers LastActionBuffers;
            public int Branch;
            public IList<int> Mask;
            public ActionSpec ActionSpec { get; }

            public void OnActionReceived(ActionBuffers actionBuffers)
            {
                LastActionBuffers = actionBuffers;
            }

            public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
            {
                actionMask.WriteMask(Branch, Mask);
            }
        }

        [Test]
        public void TestConstruct()
        {
            var ar = new TestActionReceiver();
            var va = new VectorActuator(ar, new[] { 1, 2, 3 }, SpaceType.Discrete, "name");

            Assert.IsTrue(va.ActionSpec.NumDiscreteActions == 3);
            Assert.IsTrue(va.ActionSpec.SumOfDiscreteBranchSizes == 6);
            Assert.IsTrue(va.ActionSpec.NumContinuousActions == 0);

            var va1 = new VectorActuator(ar, new[] { 4 }, SpaceType.Continuous, "name");

            Assert.IsTrue(va1.ActionSpec.NumContinuousActions == 4);
            Assert.IsTrue(va1.ActionSpec.SumOfDiscreteBranchSizes == 0);
            Assert.AreEqual(va1.Name, "name-Continuous");
        }

        [Test]
        public void TestOnActionReceived()
        {
            var ar = new TestActionReceiver();
            var va = new VectorActuator(ar, new[] { 1, 2, 3 }, SpaceType.Discrete, "name");

            var discreteActions = new[] { 0, 1, 1 };
            var ab = new ActionBuffers(ActionSegment<float>.Empty,
                new ActionSegment<int>(discreteActions, 0, 3));

            va.OnActionReceived(ab);

            Assert.AreEqual(ar.LastActionBuffers, ab);
            va.ResetData();
            Assert.AreEqual(va.ActionBuffers.ContinuousActions, ActionSegment<float>.Empty);
            Assert.AreEqual(va.ActionBuffers.DiscreteActions, ActionSegment<int>.Empty);
        }

        [Test]
        public void TestResetData()
        {
            var ar = new TestActionReceiver();
            var va = new VectorActuator(ar, new[] { 1, 2, 3 }, SpaceType.Discrete, "name");

            var discreteActions = new[] { 0, 1, 1 };
            var ab = new ActionBuffers(ActionSegment<float>.Empty,
                new ActionSegment<int>(discreteActions, 0, 3));

            va.OnActionReceived(ab);
        }

        [Test]
        public void TestWriteDiscreteActionMask()
        {
            var ar = new TestActionReceiver();
            var va = new VectorActuator(ar, new[] { 1, 2, 3 }, SpaceType.Discrete, "name");
            var bdam = new ActuatorDiscreteActionMask(new[] { va }, 6, 3);

            var groundTruthMask = new[] { false, true, false, false, true, true };

            ar.Branch = 1;
            ar.Mask = new[] { 0 };
            va.WriteDiscreteActionMask(bdam);
            ar.Branch = 2;
            ar.Mask = new[] { 1, 2 };
            va.WriteDiscreteActionMask(bdam);

            Assert.IsTrue(groundTruthMask.SequenceEqual(bdam.GetMask()));
        }
    }
}
