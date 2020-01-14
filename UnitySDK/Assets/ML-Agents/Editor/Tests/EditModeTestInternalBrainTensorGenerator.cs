using System.Collections.Generic;
using Barracuda;
using NUnit.Framework;
using UnityEngine;
using MLAgents.InferenceBrain;
using System.Reflection;


namespace MLAgents.Tests
{
    [TestFixture]
    public class EditModeTestInternalBrainTensorGenerator
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        static List<Agent> GetFakeAgents()
        {
            var goA = new GameObject("goA");
            var bpA = goA.AddComponent<BehaviorParameters>();
            bpA.brainParameters.vectorObservationSize = 3;
            bpA.brainParameters.numStackedVectorObservations = 1;
            var agentA = goA.AddComponent<TestAgent>();

            var goB = new GameObject("goB");
            var bpB = goB.AddComponent<BehaviorParameters>();
            bpB.brainParameters.vectorObservationSize = 3;
            bpB.brainParameters.numStackedVectorObservations = 1;
            var agentB = goB.AddComponent<TestAgent>();

            var agents = new List<Agent> { agentA, agentB };
            foreach (var agent in agents)
            {
                var agentEnableMethod = typeof(Agent).GetMethod("OnEnableHelper",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                agentEnableMethod?.Invoke(agent, new object[] { });
            }
            agentA.collectObservationsSensor.AddObservation(new Vector3(1, 2, 3));
            agentB.collectObservationsSensor.AddObservation(new Vector3(4, 5, 6));

            var infoA = new AgentInfo
            {
                storedVectorActions = new[] { 1f, 2f },
                actionMasks = null
            };

            var infoB = new AgentInfo
            {
                storedVectorActions = new[] { 3f, 4f },
                actionMasks = new[] { true, false, false, false, false },
            };

            agentA.Info = infoA;
            agentB.Info = infoB;
            return agents;
        }

        [Test]
        public void Construction()
        {
            var alloc = new TensorCachingAllocator();
            var mem = new Dictionary<int, List<float>>();
            var tensorGenerator = new TensorGenerator(0, alloc, mem);
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
            var inputTensor = new TensorProxy
            {
                shape = new long[] { 2, 3 }
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents();
            var alloc = new TensorCachingAllocator();
            var generator = new VectorObservationGenerator(alloc);
            generator.AddSensorIndex(0);
            generator.AddSensorIndex(1);
            generator.AddSensorIndex(2);
            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair{agentInfo = agent0.Info, sensors = agent0.sensors},
                new AgentInfoSensorsPair{agentInfo = agent1.Info, sensors = agent1.sensors},
            };
            generator.Generate(inputTensor, batchSize, inputs);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 1);
            Assert.AreEqual(inputTensor.data[0, 2], 3);
            Assert.AreEqual(inputTensor.data[1, 0], 4);
            Assert.AreEqual(inputTensor.data[1, 2], 6);
            alloc.Dispose();
        }

        [Test]
        public void GeneratePreviousActionInput()
        {
            var inputTensor = new TensorProxy
            {
                shape = new long[] { 2, 2 },
                valueType = TensorProxy.TensorType.Integer
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents();
            var alloc = new TensorCachingAllocator();
            var generator = new PreviousActionInputGenerator(alloc);
            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair{agentInfo = agent0.Info, sensors = agent0.sensors},
                new AgentInfoSensorsPair{agentInfo = agent1.Info, sensors = agent1.sensors},
            };
            generator.Generate(inputTensor, batchSize, inputs);
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
            var inputTensor = new TensorProxy
            {
                shape = new long[] { 2, 5 },
                valueType = TensorProxy.TensorType.FloatingPoint
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents();
            var alloc = new TensorCachingAllocator();
            var generator = new ActionMaskInputGenerator(alloc);

            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair{agentInfo = agent0.Info, sensors = agent0.sensors},
                new AgentInfoSensorsPair{agentInfo = agent1.Info, sensors = agent1.sensors},
            };

            generator.Generate(inputTensor, batchSize, inputs);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(inputTensor.data[0, 0], 1);
            Assert.AreEqual(inputTensor.data[0, 4], 1);
            Assert.AreEqual(inputTensor.data[1, 0], 0);
            Assert.AreEqual(inputTensor.data[1, 4], 1);
            alloc.Dispose();
        }
    }
}
