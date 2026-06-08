using NUnit.Framework;
using System;
using System.Linq;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Utils.Tests;

namespace Unity.MLAgents.Tests
{
    public class StackingSensorTests
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }

            Academy.Instance.AutomaticSteppingEnabled = false;
        }

        [TearDown]
        public void TearDown()
        {
            CommunicatorFactory.ClearCreator();
        }

        [Test]
        public void TestCtor()
        {
            ISensor wrapped = new VectorSensor(4);
            ISensor sensor = new StackingSensor(wrapped, 4);
            Assert.AreEqual("StackingSensor_size4_VectorSensor_size4", sensor.GetName());
            Assert.AreEqual(sensor.GetObservationSpec().Shape, new InplaceArray<int>(16));
        }

        [Test]
        public void AssertStackingReset()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.NumStackedVectorObservations = 3;
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            var aca = Academy.Instance;
            var agent1 = agentGo1.AddComponent<TestAgent>();
            var policy = new TestPolicy();
            agent1.SetPolicy(policy);

            StackingSensor sensor = null;
            foreach (ISensor s in agent1.sensors)
            {
                if (s is StackingSensor)
                {
                    sensor = s as StackingSensor;
                }
            }

            Assert.NotNull(sensor);

            for (int i = 0; i < 20; i++)
            {
                agent1.RequestDecision();
                aca.EnvironmentStep();
            }
            SensorTestHelper.CompareObservation(sensor, new[] { 18f, 19f, 20f });
            policy.OnRequestDecision = () => SensorTestHelper.CompareObservation(sensor, new[] { 19f, 20f, 21f });
            agent1.EndEpisode();
            policy.OnRequestDecision = () => { };
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 0f });
            for (int i = 0; i < 20; i++)
            {
                agent1.RequestDecision();
                aca.EnvironmentStep();
                SensorTestHelper.CompareObservation(sensor, new[] { Math.Max(0, i - 1f), i, i + 1 });
            }
        }

        [Test]
        public void TestVectorStacking()
        {
            VectorSensor wrapped = new VectorSensor(2);
            StackingSensor sensor = new StackingSensor(wrapped, 3);

            wrapped.AddObservation(new[] { 1f, 2f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 0f, 0f, 1f, 2f });
            var data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 0f, 0f, 0f, 0f, 1f, 2f }));

            sensor.Update();
            wrapped.AddObservation(new[] { 3f, 4f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 1f, 2f, 3f, 4f });
            data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 0f, 0f, 1f, 2f, 3f, 4f }));

            sensor.Update();
            wrapped.AddObservation(new[] { 5f, 6f });
            SensorTestHelper.CompareObservation(sensor, new[] { 1f, 2f, 3f, 4f, 5f, 6f });
            data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));

            sensor.Update();
            wrapped.AddObservation(new[] { 7f, 8f });
            SensorTestHelper.CompareObservation(sensor, new[] { 3f, 4f, 5f, 6f, 7f, 8f });
            data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 3f, 4f, 5f, 6f, 7f, 8f }));

            sensor.Update();
            wrapped.AddObservation(new[] { 9f, 10f });
            SensorTestHelper.CompareObservation(sensor, new[] { 5f, 6f, 7f, 8f, 9f, 10f });
            data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 5f, 6f, 7f, 8f, 9f, 10f }));

            // Check that if we don't call Update(), the same observations are produced
            SensorTestHelper.CompareObservation(sensor, new[] { 5f, 6f, 7f, 8f, 9f, 10f });
            data = sensor.GetStackedObservations();
            Assert.IsTrue(data.ToArray().SequenceEqual(new[] { 5f, 6f, 7f, 8f, 9f, 10f }));
        }

        [Test]
        public void TestVectorStackingReset()
        {
            VectorSensor wrapped = new VectorSensor(2);
            ISensor sensor = new StackingSensor(wrapped, 3);

            wrapped.AddObservation(new[] { 1f, 2f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 0f, 0f, 1f, 2f });

            sensor.Update();
            wrapped.AddObservation(new[] { 3f, 4f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 1f, 2f, 3f, 4f });

            sensor.Reset();
            wrapped.AddObservation(new[] { 5f, 6f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 0f, 0f, 5f, 6f });
        }

        class Dummy3DSensor : ISensor
        {
            public SensorCompressionType CompressionType = SensorCompressionType.PNG;
            public int[] Mapping;
            public ObservationSpec ObservationSpec;
            public float[,,] CurrentObservation;

            public ObservationSpec GetObservationSpec()
            {
                return ObservationSpec;
            }

            public int Write(ObservationWriter writer)
            {
                for (var c = 0; c < ObservationSpec.Shape[0]; c++)
                {
                    for (var h = 0; h < ObservationSpec.Shape[1]; h++)
                    {
                        for (var w = 0; w < ObservationSpec.Shape[2]; w++)
                        {
                            writer[c, h, w] = CurrentObservation[c, h, w];
                        }
                    }
                }
                return ObservationSpec.Shape[0] * ObservationSpec.Shape[1] * ObservationSpec.Shape[2];
            }

            public byte[] GetCompressedObservation()
            {
                var writer = new ObservationWriter();
                var flattenedObservation = new float[ObservationSpec.Shape[0] * ObservationSpec.Shape[1] * ObservationSpec.Shape[2]];
                writer.SetTarget(flattenedObservation, ObservationSpec.Shape, 0);
                Write(writer);
                byte[] bytes = Array.ConvertAll(flattenedObservation, (z) => (byte)z);
                return bytes;
            }

            public void Update() { }

            public void Reset() { }

            public CompressionSpec GetCompressionSpec()
            {
                return new CompressionSpec(CompressionType, Mapping);
            }

            public string GetName()
            {
                return "Dummy";
            }
        }

        [Test]
        public void TestStackingMapping()
        {
            // Test grayscale stacked mapping with CameraSensor
            var cameraSensor = new CameraSensor(new Camera(), 64, 64,
                true, "grayscaleCamera", SensorCompressionType.PNG);
            var stackedCameraSensor = new StackingSensor(cameraSensor, 2);
            Assert.AreEqual(stackedCameraSensor.GetCompressionSpec().CompressedChannelMapping, new[] { 0, 0, 0, 1, 1, 1 });

            // Test RGB stacked mapping with RenderTextureSensor
            var renderTextureSensor = new RenderTextureSensor(new RenderTexture(24, 16, 0),
                false, "renderTexture", SensorCompressionType.PNG);
            var stackedRenderTextureSensor = new StackingSensor(renderTextureSensor, 2);
            Assert.AreEqual(stackedRenderTextureSensor.GetCompressionSpec().CompressedChannelMapping, new[] { 0, 1, 2, 3, 4, 5 });

            // Test mapping with number of layers not being multiple of 3
            var dummySensor = new Dummy3DSensor();
            dummySensor.ObservationSpec = ObservationSpec.Visual(4, 2, 2);
            dummySensor.Mapping = new[] { 0, 1, 2, 3 };
            var stackedDummySensor = new StackingSensor(dummySensor, 2);
            Assert.AreEqual(stackedDummySensor.GetCompressionSpec().CompressedChannelMapping, new[] { 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1 });

            // Test mapping with dummy layers that should be dropped
            var paddedDummySensor = new Dummy3DSensor();
            paddedDummySensor.ObservationSpec = ObservationSpec.Visual(4, 2, 2);
            paddedDummySensor.Mapping = new[] { 0, 1, 2, 3, -1, -1 };
            var stackedPaddedDummySensor = new StackingSensor(paddedDummySensor, 2);
            Assert.AreEqual(stackedPaddedDummySensor.GetCompressionSpec().CompressedChannelMapping, new[] { 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1 });
        }

        [Test]
        public void Test3DStacking()
        {
            var wrapped = new Dummy3DSensor();
            wrapped.ObservationSpec = ObservationSpec.Visual(2, 2, 1);
            var sensor = new StackingSensor(wrapped, 2);

            // Check the stacking is on the channel dimension
            wrapped.CurrentObservation = new[, ,] { { { 1f }, { 2f } }, { { 3f }, { 4f } } };
            // var expectedObs = new[,,] { { { 0f, 0f, 1f, 2f } }, { { 0f, 0f, 3f, 4f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 0f }, { 0f } }, { { 0f }, { 0f } }, { { 1f }, { 2f } }, { { 3f }, { 4f } } });

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 5f }, { 6f } }, { { 7f }, { 8f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 1f }, { 2f } }, { { 3f }, { 4f } }, { { 5f }, { 6f } }, { { 7f }, { 8f } } });

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 9f }, { 10f } }, { { 11f }, { 12f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 5f }, { 6f } }, { { 7f }, { 8f } }, { { 9f }, { 10f } }, { { 11f }, { 12f } } });

            // Check that if we don't call Update(), the same observations are produced
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 5f }, { 6f } }, { { 7f }, { 8f } }, { { 9f }, { 10f } }, { { 11f }, { 12f } } });

            // Test reset
            sensor.Reset();
            wrapped.CurrentObservation = new[, ,] { { { 13f }, { 14f } }, { { 15f }, { 16f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 0f }, { 0f } }, { { 0f }, { 0f } }, { { 13f }, { 14f } }, { { 15f }, { 16f } } });
        }

        [Test]
        public void TestStackedGetCompressedObservation()
        {
            var wrapped = new Dummy3DSensor();
            wrapped.ObservationSpec = ObservationSpec.Visual(1, 1, 3);
            var sensor = new StackingSensor(wrapped, 2);

            wrapped.CurrentObservation = new[, ,] { { { 1f, 2f, 3f } } };
            var expected1 = sensor.CreateEmptyPNG();
            expected1 = expected1.Concat(Array.ConvertAll(new[] { 1f, 2f, 3f }, (z) => (byte)z)).ToArray();
            Assert.AreEqual(sensor.GetCompressedObservation(), expected1);

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 4f, 5f, 6f } } };
            var expected2 = Array.ConvertAll(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, (z) => (byte)z);
            Assert.AreEqual(sensor.GetCompressedObservation(), expected2);

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 7f, 8f, 9f } } };
            var expected3 = Array.ConvertAll(new[] { 4f, 5f, 6f, 7f, 8f, 9f }, (z) => (byte)z);
            Assert.AreEqual(sensor.GetCompressedObservation(), expected3);

            // Test reset
            sensor.Reset();
            wrapped.CurrentObservation = new[, ,] { { { 10f, 11f, 12f } } };
            var expected4 = sensor.CreateEmptyPNG();
            expected4 = expected4.Concat(Array.ConvertAll(new[] { 10f, 11f, 12f }, (z) => (byte)z)).ToArray();
            Assert.AreEqual(sensor.GetCompressedObservation(), expected4);
        }

        [Test]
        public void TestStackingSensorBuiltInSensorType()
        {
            var dummySensor = new Dummy3DSensor();
            dummySensor.ObservationSpec = ObservationSpec.Visual(2, 2, 4);
            dummySensor.Mapping = new[] { 0, 1, 2, 3 };
            var stackedDummySensor = new StackingSensor(dummySensor, 2);
            Assert.AreEqual(stackedDummySensor.GetBuiltInSensorType(), BuiltInSensorType.Unknown);

            var vectorSensor = new VectorSensor(4);
            var stackedVectorSensor = new StackingSensor(vectorSensor, 4);
            Assert.AreEqual(stackedVectorSensor.GetBuiltInSensorType(), BuiltInSensorType.VectorSensor);
        }
    }
}
