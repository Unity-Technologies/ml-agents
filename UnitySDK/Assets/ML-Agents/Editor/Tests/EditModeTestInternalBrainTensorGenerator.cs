using System;
using System.Collections.Generic;
using System.Linq;
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
            var tensorGenerator = new TensorGenerator(bp, 0);
            Assert.IsNotNull(tensorGenerator);
        }

        [Test]
        public void GenerateBatchSize()
        {
            var inputTensor = new Tensor();
            var batchSize = 4;
            var generator = new BatchSizeGenerator();
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data as int[]);
            Assert.AreEqual((inputTensor.Data as int[])[0], batchSize);
        }
        
        [Test]
        public void GenerateSequenceLength()
        {
            var inputTensor = new Tensor();
            var batchSize = 4;
            var generator = new SequenceLengthGenerator();
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data as int[]);
            Assert.AreEqual((inputTensor.Data as int[])[0], 1);
        }
        
        [Test]
        public void GenerateVectorObservation()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 3}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            
            var generator = new VectorObservationGenerator();
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 2], 3);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 4);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 2], 6);
        }
        
        [Test]
        public void GenerateRecurrentInput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            
            var generator = new RecurrentInputGenerator();
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 4], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 4], 0);
        }
        
        [Test]
        public void GeneratePreviousActionInput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 2},
                ValueType = Tensor.TensorType.Integer
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();

            var generator = new PreviousActionInputGenerator();

            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as int[,]);
            Assert.AreEqual((inputTensor.Data as int[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as int[,])[0, 1], 2);
            Assert.AreEqual((inputTensor.Data as int[,])[1, 0], 3);
            Assert.AreEqual((inputTensor.Data as int[,])[1, 1], 4);
        }
        
        [Test]
        public void GenerateActionMaskInput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5},
                ValueType = Tensor.TensorType.FloatingPoint
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
  
            var generator = new ActionMaskInputGenerator();
            generator.Generate(inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 4], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 4], 1);
        }
    }
}
