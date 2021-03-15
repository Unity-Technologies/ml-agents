using System.Collections.Generic;
using Unity.Barracuda;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;

namespace Unity.MLAgents.Tests
{

    public class DiscreteActionOutputApplierTest
    {
        [Test]
        public void TestDiscreteApply()
        {
            var actionSpec = ActionSpec.MakeDiscrete(3, 2);

            var applier = new DiscreteActionOutputApplier(actionSpec, 2020, null);
            var agentIds = new List<int> { 42, 1337 };
            var actionBuffers = new Dictionary<int, ActionBuffers>();
            actionBuffers[42] = new ActionBuffers(actionSpec);
            actionBuffers[1337] = new ActionBuffers(actionSpec);

            var actionTensor = new TensorProxy
            {
                data = new Tensor(
                    2,
                    2,
                    new[]
                    {
                        2.0f, // Agent 0, branch 0
                        1.0f, // Agent 0, branch 1
                        0.0f, // Agent 1, branch 0
                        0.0f  // Agent 1, branch 1
                    }),
                shape = new long[] { 2, 2 },
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            applier.Apply(actionTensor, agentIds, actionBuffers);
            Assert.AreEqual(2, actionBuffers[42].DiscreteActions[0]);
            Assert.AreEqual(1, actionBuffers[42].DiscreteActions[1]);

            Assert.AreEqual(0, actionBuffers[1337].DiscreteActions[0]);
            Assert.AreEqual(0, actionBuffers[1337].DiscreteActions[1]);
        }
    }

    public class LegacyDiscreteActionOutputApplierTest
    {
        [Test]
        public void TestDiscreteApply()
        {
            var actionSpec = ActionSpec.MakeDiscrete(3, 2);
            const float smallLogProb = -1000.0f;
            const float largeLogProb = -1.0f;

            var logProbs = new TensorProxy
            {
                data = new Tensor(
                    2,
                    5,
                    new[]
                    {
                        smallLogProb, smallLogProb, largeLogProb, // Agent 0, branch 0
                        smallLogProb, largeLogProb,               // Agent 0, branch 1
                        largeLogProb, smallLogProb, smallLogProb, // Agent 1, branch 0
                        largeLogProb, smallLogProb,               // Agent 1, branch 1
                    }),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var applier = new LegacyDiscreteActionOutputApplier(actionSpec, 2020, null);
            var agentIds = new List<int> { 42, 1337 };
            var actionBuffers = new Dictionary<int, ActionBuffers>();
            actionBuffers[42] = new ActionBuffers(actionSpec);
            actionBuffers[1337] = new ActionBuffers(actionSpec);

            applier.Apply(logProbs, agentIds, actionBuffers);
            Assert.AreEqual(2, actionBuffers[42].DiscreteActions[0]);
            Assert.AreEqual(1, actionBuffers[42].DiscreteActions[1]);

            Assert.AreEqual(0, actionBuffers[1337].DiscreteActions[0]);
            Assert.AreEqual(0, actionBuffers[1337].DiscreteActions[1]);
        }
    }
}
