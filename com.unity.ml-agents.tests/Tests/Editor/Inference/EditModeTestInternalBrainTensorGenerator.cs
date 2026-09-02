using System.Collections.Generic;
using Unity.InferenceEngine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Utils.Tests;
#if !UNITY_EDITOR_WIN

namespace Unity.MLAgents.Tests
{
    internal class OverflowSensor : ISensor
    {
        readonly string m_Name;
        readonly int m_Size;

        public OverflowSensor(string name, int size)
        {
            m_Name = name;
            m_Size = size;
        }

        public ObservationSpec GetObservationSpec()
        {
            return ObservationSpec.Vector(m_Size);
        }

        public int Write(ObservationWriter writer)
        {
            for (var i = 0; i < m_Size; i++)
                writer[i] = i + 1f;
            return m_Size;
        }

        public byte[] GetCompressedObservation() { return null; }
        public CompressionSpec GetCompressionSpec() { return new CompressionSpec(SensorCompressionType.None); }
        public string GetName() { return m_Name; }
        public void Update() { }
        public void Reset() { }
    }

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

        static List<TestAgent> GetFakeAgents(ObservableAttributeOptions observableAttributeOptions = ObservableAttributeOptions.Ignore)
        {
            var goA = new GameObject("goA");
            var bpA = goA.AddComponent<BehaviorParameters>();
            bpA.BrainParameters.VectorObservationSize = 3;
            bpA.BrainParameters.NumStackedVectorObservations = 1;
            bpA.ObservableAttributeHandling = observableAttributeOptions;
            var agentA = goA.AddComponent<TestAgent>();

            var goB = new GameObject("goB");
            var bpB = goB.AddComponent<BehaviorParameters>();
            bpB.BrainParameters.VectorObservationSize = 3;
            bpB.BrainParameters.NumStackedVectorObservations = 1;
            bpB.ObservableAttributeHandling = observableAttributeOptions;
            var agentB = goB.AddComponent<TestAgent>();

            var agents = new List<TestAgent> { agentA, agentB };
            foreach (var agent in agents)
            {
                agent.LazyInitialize();
            }
            agentA.collectObservationsSensor.AddObservation(new Vector3(1, 2, 3));
            agentB.collectObservationsSensor.AddObservation(new Vector3(4, 5, 6));

            var infoA = new AgentInfo
            {
                storedActions = new ActionBuffers(null, new[] { 1, 2 }),
                discreteActionMasks = null,
            };

            var infoB = new AgentInfo
            {
                storedActions = new ActionBuffers(null, new[] { 3, 4 }),
                discreteActionMasks = new[] { true, false, false, false, false },
            };


            agentA._Info = infoA;
            agentB._Info = infoB;
            return agents;
        }

        [Test]
        public void Construction()
        {
            var mem = new Dictionary<int, List<float>>();
            var tensorGenerator = new TensorGenerator(0, mem);
            Assert.IsNotNull(tensorGenerator);
        }

        [Test]
        public void GenerateBatchSize()
        {
            var inputTensor = new TensorProxy();
            const int batchSize = 4;
            var generator = new BatchSizeGenerator();
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[0], batchSize);
        }

        [Test]
        public void GenerateSequenceLength()
        {
            var inputTensor = new TensorProxy();
            const int batchSize = 4;
            var generator = new SequenceLengthGenerator();
            generator.Generate(inputTensor, batchSize, null);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[0], 1);
        }

        [Test]
        public void GenerateVectorObservation()
        {
            var inputTensor = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                shape = new int[] { 2, 4 }
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents(ObservableAttributeOptions.ExamineAll);
            var generator = new ObservationGenerator();
            generator.AddSensorIndex(0); // ObservableAttribute (size 1)
            generator.AddSensorIndex(1); // TestSensor (size 0)
            generator.AddSensorIndex(2); // TestSensor (size 0)
            generator.AddSensorIndex(3); // VectorSensor (size 3)
            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair { agentInfo = agent0._Info, sensors = agent0.sensors },
                new AgentInfoSensorsPair { agentInfo = agent1._Info, sensors = agent1.sensors },
            };
            generator.Generate(inputTensor, batchSize, inputs);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[0, 1], 1);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[0, 3], 3);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[1, 1], 4);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[1, 3], 6);
        }

        [Test]
        public void GeneratePreviousActionInput()
        {
            var inputTensor = new TensorProxy
            {
                shape = new int[] { 2, 2 },
                valueType = TensorProxy.TensorType.Integer
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents();
            var generator = new PreviousActionInputGenerator();
            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair { agentInfo = agent0._Info, sensors = agent0.sensors },
                new AgentInfoSensorsPair { agentInfo = agent1._Info, sensors = agent1.sensors },
            };
            generator.Generate(inputTensor, batchSize, inputs);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[0, 0], 1);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[0, 1], 2);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[1, 0], 3);
            Assert.AreEqual(((Tensor<int>)inputTensor.data)[1, 1], 4);
        }

        [Test]
        public void GenerateActionMaskInput()
        {
            var inputTensor = new TensorProxy
            {
                shape = new int[] { 2, 5 },
                valueType = TensorProxy.TensorType.FloatingPoint
            };
            const int batchSize = 4;
            var agentInfos = GetFakeAgents();
            var generator = new ActionMaskInputGenerator();

            var agent0 = agentInfos[0];
            var agent1 = agentInfos[1];
            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair { agentInfo = agent0._Info, sensors = agent0.sensors },
                new AgentInfoSensorsPair { agentInfo = agent1._Info, sensors = agent1.sensors },
            };

            generator.Generate(inputTensor, batchSize, inputs);
            Assert.IsNotNull(inputTensor.data);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[0, 0], 1);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[0, 4], 1);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[1, 0], 0);
            Assert.AreEqual((int)((Tensor<float>)inputTensor.data)[1, 4], 1);
        }

        [Test]
        public void GenerateVectorObservation_CapacityGuardPreventsOverflow()
        {
            // Tensor can hold 3 floats, sensor0 fills it exactly (3),
            // so the guard fires before sensor1 can write anything.
            var inputTensor = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                shape = new int[] { 1, 3 }
            };

            var sensor0 = new OverflowSensor("sensor0", 3);
            var sensor1 = new OverflowSensor("sensor1", 3);
            var sensors = new List<ISensor> { sensor0, sensor1 };

            var generator = new ObservationGenerator();
            generator.AddSensorIndex(0);
            generator.AddSensorIndex(1);

            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair
                {
                    agentInfo = new AgentInfo { done = false },
                    sensors = sensors
                }
            };

            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex("Sensor write overflow"));
            generator.Generate(inputTensor, 1, inputs);

            // First sensor's 3 writes land, second sensor is skipped by capacity guard
            Assert.AreEqual(1f, ((Tensor<float>)inputTensor.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)inputTensor.data)[0, 1]);
            Assert.AreEqual(3f, ((Tensor<float>)inputTensor.data)[0, 2]);
        }

        [Test]
        public void GenerateVectorObservation_SingleSensorOverflowIsClamped()
        {
            // Tensor can hold 2 floats, but sensor writes 5
            var inputTensor = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                shape = new int[] { 1, 2 }
            };

            var sensor = new OverflowSensor("big_sensor", 5);
            var sensors = new List<ISensor> { sensor };

            var generator = new ObservationGenerator();
            generator.AddSensorIndex(0);

            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair
                {
                    agentInfo = new AgentInfo { done = false },
                    sensors = sensors
                }
            };

            // No crash — ObservationWriter bounds cap prevents the buffer overrun
            generator.Generate(inputTensor, 1, inputs);

            Assert.AreEqual(1f, ((Tensor<float>)inputTensor.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)inputTensor.data)[0, 1]);
        }

        [Test]
        public void GenerateVectorObservation_ExactFitNoWarning()
        {
            // Tensor exactly fits the sensor output — no warning should fire
            var inputTensor = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                shape = new int[] { 1, 3 }
            };

            var sensor = new OverflowSensor("exact_sensor", 3);
            var sensors = new List<ISensor> { sensor };

            var generator = new ObservationGenerator();
            generator.AddSensorIndex(0);

            var inputs = new List<AgentInfoSensorsPair>
            {
                new AgentInfoSensorsPair
                {
                    agentInfo = new AgentInfo { done = false },
                    sensors = sensors
                }
            };

            generator.Generate(inputTensor, 1, inputs);

            Assert.AreEqual(1f, ((Tensor<float>)inputTensor.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)inputTensor.data)[0, 1]);
            Assert.AreEqual(3f, ((Tensor<float>)inputTensor.data)[0, 2]);
        }
    }
}
#endif
