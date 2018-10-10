using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;
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
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            Assert.IsNotNull(tensorGenerator);
        }

        [Test]
        public void GenerateBatchSize()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            var inputTensor = new Tensor();
            var batchSize = 4;
            tensorGenerator[TensorNames.BatchSizePlaceholder].Execute(
                inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data as int[]);
            Assert.AreEqual((inputTensor.Data as int[])[0], batchSize);
        }
        
        [Test]
        public void GenerateSequenceLength()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            var inputTensor = new Tensor();
            var batchSize = 4;
            tensorGenerator[TensorNames.SequenceLengthPlaceholder].Execute(
                inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.Data as int[]);
            Assert.AreEqual((inputTensor.Data as int[])[0], 1);
        }
        
        [Test]
        public void GenerateVectorObservation()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 3}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            
            tensorGenerator[TensorNames.VectorObservationPlacholder].Execute(
                inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 2], 3);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 4);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 2], 6);
        }
        
        [Test]
        public void GenerateRecurrentInput()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5}
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            
            tensorGenerator[TensorNames.RecurrentInPlaceholder].Execute(
                inputTensor, batchSize, agentInfos);
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 4], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 4], 0);
        }
        
        [Test]
        public void GeneratePreviousActionInput()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp,0);
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 2},
                ValueType = Tensor.TensorType.FloatingPoint
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
            
            Assert.Catch<NotImplementedException>(
                () => tensorGenerator[TensorNames.PreviousActionPlaceholder].Execute(
                    inputTensor, batchSize, agentInfos));

            inputTensor.ValueType = Tensor.TensorType.Integer;
            tensorGenerator[TensorNames.PreviousActionPlaceholder].Execute(
                inputTensor, batchSize, agentInfos);
            
            Assert.IsNotNull(inputTensor.Data as int[,]);
            Assert.AreEqual((inputTensor.Data as int[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as int[,])[0, 1], 2);
            Assert.AreEqual((inputTensor.Data as int[,])[1, 0], 3);
            Assert.AreEqual((inputTensor.Data as int[,])[1, 1], 4);
        }
        
        [Test]
        public void GenerateActionMaskInput()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorGeneratorInvoker(bp, 0);
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5},
                ValueType = Tensor.TensorType.FloatingPoint
                
            };
            var batchSize = 4;
            var agentInfos = GetFakeAgentInfos();
  
            tensorGenerator[TensorNames.ActionMaskPlaceholder].Execute(
                inputTensor, batchSize, agentInfos);
            
            Assert.IsNotNull(inputTensor.Data as float[,]);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 0], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[0, 4], 1);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 0], 0);
            Assert.AreEqual((inputTensor.Data as float[,])[1, 4], 1);
        }
    }
}
