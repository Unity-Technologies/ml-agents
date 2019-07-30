using System;
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
                stackedVectorObservation = (new float[] {1f, 2f, 3f}).ToList(),
                memories = null,
                storedVectorActions = new float[] {1, 2},
                actionMasks = null,
                
            };
            var goB = new GameObject("goB");
            var agentB = goB.AddComponent<TestAgent>();
            var infoB = new AgentInfo()
            {
                stackedVectorObservation = (new float[] {4f, 5f, 6f}).ToList(),
                memories = (new float[] {1f, 1f, 1f}).ToList(),
                storedVectorActions = new float[] {3, 4},
                actionMasks = new bool[] {true, false, false, false, false},
            };

            return new Dictionary<Agent, AgentInfo>(){{agentA, infoA},{agentB, infoB}};
        }

        [Test]
        public void Contruction()
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
            var batchSize = 4;
            var generator = new BatchSizeGenerator(alloc);
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0], batchSize);
            alloc.Dispose();
        }
        
        [Test]
        public void GenerateSequenceLength()
        {
            var inputTensor = new TensorProxy();
            var alloc = new TensorCachingAllocator();
            var batchSize = 4;
            var generator = new SequenceLengthGenerator(alloc);
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0], 1);
            alloc.Dispose();
        }
        
        [Test]
        public void GenerateVectorObservation()
        {
            var inputTensor = new TensorProxy()
            {
                Shape = new long[] {2, 3}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new VectorObservationGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0, 0], 1);
            Assert.AreEqual(inputTensor.Data[0, 2], 3);
            Assert.AreEqual(inputTensor.Data[1, 0], 4);
            Assert.AreEqual(inputTensor.Data[1, 2], 6);
            alloc.Dispose();
        }
        
        [Test]
        public void GenerateRecurrentInput()
        {
            var inputTensor = new TensorProxy()
            {
                Shape = new long[] {2, 5}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new RecurrentInputGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0, 0], 0);
            Assert.AreEqual(inputTensor.Data[0, 4], 0);
            Assert.AreEqual(inputTensor.Data[1, 0], 1);
            Assert.AreEqual(inputTensor.Data[1, 4], 0);
            alloc.Dispose();
        }
        
        [Test]
        public void GeneratePreviousActionInput()
        {
            var inputTensor = new TensorProxy()
            {
                Shape = new long[] {2, 2},
                ValueType = TensorProxy.TensorType.Integer
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new PreviousActionInputGenerator(alloc);

            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0, 0], 1);
            Assert.AreEqual(inputTensor.Data[0, 1], 2);
            Assert.AreEqual(inputTensor.Data[1, 0], 3);
            Assert.AreEqual(inputTensor.Data[1, 1], 4);
            alloc.Dispose();
        }
        
        [Test]
        public void GenerateActionMaskInput()
        {
            var inputTensor = new TensorProxy()
            {
                Shape = new long[] {2, 5},
                ValueType = TensorProxy.TensorType.FloatingPoint
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var generator = new ActionMaskInputGenerator(alloc);
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data);
            Assert.AreEqual(inputTensor.Data[0, 0], 1);
            Assert.AreEqual(inputTensor.Data[0, 4], 1);
            Assert.AreEqual(inputTensor.Data[1, 0], 0);
            Assert.AreEqual(inputTensor.Data[1, 4], 1);
            alloc.Dispose();
        }
    }
}
