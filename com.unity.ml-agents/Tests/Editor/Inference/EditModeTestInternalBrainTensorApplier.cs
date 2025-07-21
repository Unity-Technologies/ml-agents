using System.Collections.Generic;
using NUnit.Framework;
using Unity.InferenceEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;

namespace Unity.MLAgents.Tests
{
    public class EditModeTestInternalBrainTensorApplier
    {
        class TestAgent : Agent { }

        [Test]
        public void Construction()
        {
            var actionSpec = new ActionSpec();
            var mem = new Dictionary<int, List<float>>();
            var tensorGenerator = new TensorApplier(actionSpec, 0, mem);
            Assert.IsNotNull(tensorGenerator);
        }

        [Test]
        public void ApplyContinuousActionOutput()
        {
            var actionSpec = ActionSpec.MakeContinuous(3);
            var inputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 3 },
                data = new Tensor<float>(new TensorShape(2, 3), new float[] { 1, 2, 3, 4, 5, 6 })
            };

            var applier = new ContinuousActionOutputApplier(actionSpec);

            var agentIds = new List<int>() { 0, 1 };

            // Dictionary from AgentId to Action
            var actionDict = new Dictionary<int, ActionBuffers>() { { 0, ActionBuffers.Empty }, { 1, ActionBuffers.Empty } };

            applier.Apply(inputTensor, agentIds, actionDict);


            Assert.AreEqual(actionDict[0].ContinuousActions[0], 1);
            Assert.AreEqual(actionDict[0].ContinuousActions[1], 2);
            Assert.AreEqual(actionDict[0].ContinuousActions[2], 3);

            Assert.AreEqual(actionDict[1].ContinuousActions[0], 4);
            Assert.AreEqual(actionDict[1].ContinuousActions[1], 5);
            Assert.AreEqual(actionDict[1].ContinuousActions[2], 6);
        }

        [Test]
        public void ApplyDiscreteActionOutputLegacy()
        {
            var actionSpec = ActionSpec.MakeDiscrete(2, 3);
            var inputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 5 },
                data = new Tensor<float>(
                    new TensorShape(2, 5),
                    new[] { 0.5f, 22.5f, 0.1f, 5f, 1f, 4f, 5f, 6f, 7f, 8f })
            };
            var applier = new LegacyDiscreteActionOutputApplier(actionSpec, 0);

            var agentIds = new List<int>() { 0, 1 };

            // Dictionary from AgentId to Action
            var actionDict = new Dictionary<int, ActionBuffers>() { { 0, ActionBuffers.Empty }, { 1, ActionBuffers.Empty } };


            applier.Apply(inputTensor, agentIds, actionDict);

            Assert.AreEqual(actionDict[0].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[0].DiscreteActions[1], 1);

            Assert.AreEqual(actionDict[1].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[1].DiscreteActions[1], 2);
        }

        [Test]
        public void ApplyDiscreteActionOutput()
        {
            var actionSpec = ActionSpec.MakeDiscrete(2, 3);
            var inputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 2 },
                data = new Tensor<int>(
                    new TensorShape(2, 2),
                    new[] { 1, 1, 1, 2 }),
                valueType = TensorProxy.TensorType.Integer
            };
            var applier = new DiscreteActionOutputApplier(actionSpec, 0);

            var agentIds = new List<int>() { 0, 1 };

            // Dictionary from AgentId to Action
            var actionDict = new Dictionary<int, ActionBuffers>() { { 0, ActionBuffers.Empty }, { 1, ActionBuffers.Empty } };


            applier.Apply(inputTensor, agentIds, actionDict);

            Assert.AreEqual(actionDict[0].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[0].DiscreteActions[1], 1);

            Assert.AreEqual(actionDict[1].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[1].DiscreteActions[1], 2);
        }

        [Test]
        public void ApplyHybridActionOutputLegacy()
        {
            var actionSpec = new ActionSpec(3, new[] { 2, 3 });
            var continuousInputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 3 },
                data = new Tensor<float>(new TensorShape(2, 3), new float[] { 1, 2, 3, 4, 5, 6 })
            };
            var discreteInputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 8 },
                data = new Tensor<float>(
                    new TensorShape(2, 5),
                    new[] { 0.5f, 22.5f, 0.1f, 5f, 1f, 4f, 5f, 6f, 7f, 8f })
            };
            var continuousApplier = new ContinuousActionOutputApplier(actionSpec);
            var discreteApplier = new LegacyDiscreteActionOutputApplier(actionSpec, 0);

            var agentIds = new List<int>() { 0, 1 };

            // Dictionary from AgentId to Action
            var actionDict = new Dictionary<int, ActionBuffers>() { { 0, ActionBuffers.Empty }, { 1, ActionBuffers.Empty } };


            continuousApplier.Apply(continuousInputTensor, agentIds, actionDict);
            discreteApplier.Apply(discreteInputTensor, agentIds, actionDict);

            Assert.AreEqual(actionDict[0].ContinuousActions[0], 1);
            Assert.AreEqual(actionDict[0].ContinuousActions[1], 2);
            Assert.AreEqual(actionDict[0].ContinuousActions[2], 3);
            Assert.AreEqual(actionDict[0].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[0].DiscreteActions[1], 1);

            Assert.AreEqual(actionDict[1].ContinuousActions[0], 4);
            Assert.AreEqual(actionDict[1].ContinuousActions[1], 5);
            Assert.AreEqual(actionDict[1].ContinuousActions[2], 6);
            Assert.AreEqual(actionDict[1].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[1].DiscreteActions[1], 2);
        }

        [Test]
        public void ApplyHybridActionOutput()
        {
            var actionSpec = new ActionSpec(3, new[] { 2, 3 });
            var continuousInputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 3 },
                data = new Tensor<float>(new TensorShape(2, 3), new float[] { 1, 2, 3, 4, 5, 6 }),
                valueType = TensorProxy.TensorType.FloatingPoint
            };
            var discreteInputTensor = new TensorProxy()
            {
                shape = new int[] { 2, 2 },
                data = new Tensor<int>(
                    new TensorShape(2, 2),
                    new[] { 1, 1, 1, 2 }),
                valueType = TensorProxy.TensorType.Integer
            };
            var continuousApplier = new ContinuousActionOutputApplier(actionSpec);
            var discreteApplier = new DiscreteActionOutputApplier(actionSpec, 0);

            var agentIds = new List<int>() { 0, 1 };

            // Dictionary from AgentId to Action
            var actionDict = new Dictionary<int, ActionBuffers>() { { 0, ActionBuffers.Empty }, { 1, ActionBuffers.Empty } };


            continuousApplier.Apply(continuousInputTensor, agentIds, actionDict);
            discreteApplier.Apply(discreteInputTensor, agentIds, actionDict);

            Assert.AreEqual(actionDict[0].ContinuousActions[0], 1);
            Assert.AreEqual(actionDict[0].ContinuousActions[1], 2);
            Assert.AreEqual(actionDict[0].ContinuousActions[2], 3);
            Assert.AreEqual(actionDict[0].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[0].DiscreteActions[1], 1);

            Assert.AreEqual(actionDict[1].ContinuousActions[0], 4);
            Assert.AreEqual(actionDict[1].ContinuousActions[1], 5);
            Assert.AreEqual(actionDict[1].ContinuousActions[2], 6);
            Assert.AreEqual(actionDict[1].DiscreteActions[0], 1);
            Assert.AreEqual(actionDict[1].DiscreteActions[1], 2);
        }
    }
}
