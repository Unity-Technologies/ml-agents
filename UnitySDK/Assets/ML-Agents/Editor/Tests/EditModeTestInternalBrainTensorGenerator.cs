using System.Collections.Generic;
using System.Linq;
using Barracuda;
using NUnit.Framework;
using UnityEngine;
using MLAgents.InferenceBrain;

namespace MLAgents.Tests
{
    public class EditModeTestInternalBrainTensorGenerator
    {
        private class TestAgent : Agent
        {
        }

        private Dictionary<Agent, AgentInfo> GetFakeAgentInfos()
        {
            var goA = new GameObject("goA");
            var agentA = goA.AddComponent<TestAgent>();
            var infoA = new AgentInfo()
            {
                stackedVectorObservation = (new[] {1f, 2f, 3f}).ToList(),
                memories = null,
                storedVectorActions = new[] {1f, 2f},
                actionMasks = null,
            };
            var goB = new GameObject("goB");
            var agentB = goB.AddComponent<TestAgent>();
            var infoB = new AgentInfo()
            {
                stackedVectorObservation = (new[] {4f, 5f, 6f}).ToList(),
                memories = (new[] {1f, 1f, 1f}).ToList(),
                storedVectorActions = new[] {3f, 4f},
                actionMasks = new[] {true, false, false, false, false},
            };

            return new Dictionary<Agent, AgentInfo>(){{agentA, infoA}, {agentB, infoB}};
        }

        [Test]
        public void Construction()
        {
            var bp = new BrainParameters();
            var alloc = new TensorCachingAllocator();
            var tensorGenerator = new TensorGenerator(bp, 0, alloc);
            Assert.IsNotNull(tensorGenerator);
            alloc.Dispose();
        }

        [Test]
        public void GenerateBatchSize()
        {
            var inputTensor = new TensorProxy();
            var alloc = new TensorCachingAllocator();
            const int batchSize = 4;
            var generator = new BatchSizeGenerator(alloc);
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0], batchSize);
            alloc.Dispose();
        }

        [Test]
        public void GenerateSequenceLength()
        {
            var inputTensor = new TensorProxy();
            var alloc = new TensorCachingAllocator();
            const int batchSize = 4;
            var generator = new SequenceLengthGenerator(alloc);
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0], 1);
            alloc.Dispose();
        }

        [Test]
        public void GenerateVectorObservation()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] {2, 3}
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new VectorObservationGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 1);
            Assert.AreEqual(inputTensor.data[0, 2], 3);
            Assert.AreEqual(inputTensor.data[1, 0], 4);
            Assert.AreEqual(inputTensor.data[1, 2], 6);
            alloc.Dispose();
        }

        [Test]
        public void GenerateRecurrentInput()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] {2, 5}
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new RecurrentInputGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 0);
            Assert.AreEqual(inputTensor.data[0, 4], 0);
            Assert.AreEqual(inputTensor.data[1, 0], 1);
            Assert.AreEqual(inputTensor.data[1, 4], 0);
            alloc.Dispose();
        }

        [Test]
        public void GeneratePreviousActionInput()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] {2, 2},
                valueType = TensorProxy.TensorType.Integer
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new PreviousActionInputGenerator(alloc);

            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 1);
            Assert.AreEqual(inputTensor.data[0, 1], 2);
            Assert.AreEqual(inputTensor.data[1, 0], 3);
            Assert.AreEqual(inputTensor.data[1, 1], 4);
            alloc.Dispose();
        }

        [Test]
        public void GenerateActionMaskInput()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] {2, 5},
                valueType = TensorProxy.TensorType.FloatingPoint
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new ActionMaskInputGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 1);
            Assert.AreEqual(inputTensor.data[0, 4], 1);
            Assert.AreEqual(inputTensor.data[1, 0], 0);
            Assert.AreEqual(inputTensor.data[1, 4], 1);
            alloc.Dispose();
        }
    }
}
