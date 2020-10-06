using NUnit.Framework;
using System;
using System.Linq;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    public class StackingSensorTests
    {
        [Test]
        public void TestCtor()
        {
            ISensor wrapped = new VectorSensor(4);
            ISensor sensor = new StackingSensor(wrapped, 4);
            Assert.AreEqual("StackingSensor_size4_VectorSensor_size4", sensor.GetName());
            Assert.AreEqual(sensor.GetObservationShape(), new[] { 16 });
        }

        [Test]
        public void TestVectorStacking()
        {
            VectorSensor wrapped = new VectorSensor(2);
            ISensor sensor = new StackingSensor(wrapped, 3);

            wrapped.AddObservation(new[] { 1f, 2f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 0f, 0f, 1f, 2f });

            sensor.Update();
            wrapped.AddObservation(new[] { 3f, 4f });
            SensorTestHelper.CompareObservation(sensor, new[] { 0f, 0f, 1f, 2f, 3f, 4f });

            sensor.Update();
            wrapped.AddObservation(new[] { 5f, 6f });
            SensorTestHelper.CompareObservation(sensor, new[] { 1f, 2f, 3f, 4f, 5f, 6f });

            sensor.Update();
            wrapped.AddObservation(new[] { 7f, 8f });
            SensorTestHelper.CompareObservation(sensor, new[] { 3f, 4f, 5f, 6f, 7f, 8f });

            sensor.Update();
            wrapped.AddObservation(new[] { 9f, 10f });
            SensorTestHelper.CompareObservation(sensor, new[] { 5f, 6f, 7f, 8f, 9f, 10f });

            // Check that if we don't call Update(), the same observations are produced
            SensorTestHelper.CompareObservation(sensor, new[] { 5f, 6f, 7f, 8f, 9f, 10f });
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

        class Dummy3DSensor : ISparseChannelSensor
        {
            public SensorCompressionType CompressionType = SensorCompressionType.PNG;
            public int[] Mapping;
            public int[] Shape;
            public float[,,] CurrentObservation;

            internal Dummy3DSensor()
            {
            }

            public int[] GetObservationShape()
            {
                return Shape;
            }

            public int Write(ObservationWriter writer)
            {
                for (var h = 0; h < Shape[0]; h++)
                {
                    for (var w = 0; w < Shape[1]; w++)
                    {
                        for (var c = 0; c < Shape[2]; c++)
                        {
                            writer[h, w, c] = CurrentObservation[h, w, c];
                        }
                    }
                }
                return Shape[0] * Shape[1] * Shape[2];
            }

            public byte[] GetCompressedObservation()
            {
                var writer = new ObservationWriter();
                var flattenedObservation = new float[Shape[0] * Shape[1] * Shape[2]];
                writer.SetTarget(flattenedObservation, Shape, 0);
                Write(writer);
                byte[] bytes = Array.ConvertAll(flattenedObservation, (z) => (byte)z);
                return bytes;
            }

            public void Update() { }

            public void Reset() { }

            public SensorCompressionType GetCompressionType()
            {
                return CompressionType;
            }

            public string GetName()
            {
                return "Dummy";
            }

            public int[] GetCompressedChannelMapping()
            {
                return Mapping;
            }
        }

        [Test]
        public void TestStackingMapping()
        {
            // Test grayscale stacked mapping with CameraSensor
            var cameraSensor = new CameraSensor(new Camera(), 64, 64,
                true, "grayscaleCamera", SensorCompressionType.PNG);
            var stackedCameraSensor = new StackingSensor(cameraSensor, 2);
            Assert.AreEqual(stackedCameraSensor.GetCompressedChannelMapping(), new[] { 0, 0, 0, 1, 1, 1 });

            // Test RGB stacked mapping with RenderTextureSensor
            var renderTextureSensor = new RenderTextureSensor(new RenderTexture(24, 16, 0),
                false, "renderTexture", SensorCompressionType.PNG);
            var stackedRenderTextureSensor = new StackingSensor(renderTextureSensor, 2);
            Assert.AreEqual(stackedRenderTextureSensor.GetCompressedChannelMapping(), new[] { 0, 1, 2, 3, 4, 5 });

            // Test mapping with number of layers not being multiple of 3
            var dummySensor = new Dummy3DSensor();
            dummySensor.Shape = new int[] { 2, 2, 4 };
            dummySensor.Mapping = new int[] { 0, 1, 2, 3 };
            var stackedDummySensor = new StackingSensor(dummySensor, 2);
            Assert.AreEqual(stackedDummySensor.GetCompressedChannelMapping(), new[] { 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1 });

            // Test mapping with dummy layers that should be dropped
            var paddedDummySensor = new Dummy3DSensor();
            paddedDummySensor.Shape = new int[] { 2, 2, 4 };
            paddedDummySensor.Mapping = new int[] { 0, 1, 2, 3, -1, -1 };
            var stackedPaddedDummySensor = new StackingSensor(paddedDummySensor, 2);
            Assert.AreEqual(stackedPaddedDummySensor.GetCompressedChannelMapping(), new[] { 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1 });
        }

        [Test]
        public void Test3DStacking()
        {
            var wrapped = new Dummy3DSensor();
            wrapped.Shape = new int[] { 2, 1, 2 };
            var sensor = new StackingSensor(wrapped, 2);

            // Check the stacking is on the last dimension
            wrapped.CurrentObservation = new[, ,] { { { 1f, 2f } }, { { 3f, 4f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 0f, 0f, 1f, 2f } }, { { 0f, 0f, 3f, 4f } } });

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 5f, 6f } }, { { 7f, 8f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 1f, 2f, 5f, 6f } }, { { 3f, 4f, 7f, 8f } } });

            sensor.Update();
            wrapped.CurrentObservation = new[, ,] { { { 9f, 10f } }, { { 11f, 12f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 5f, 6f, 9f, 10f } }, { { 7f, 8f, 11f, 12f } } });

            // Check that if we don't call Update(), the same observations are produced
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 5f, 6f, 9f, 10f } }, { { 7f, 8f, 11f, 12f } } });

            // Test reset
            sensor.Reset();
            wrapped.CurrentObservation = new[, ,] { { { 13f, 14f } }, { { 15f, 16f } } };
            SensorTestHelper.CompareObservation(sensor, new[, ,] { { { 0f, 0f, 13f, 14f } }, { { 0f, 0f, 15f, 16f } } });
        }

        [Test]
        public void TestStackedGetCompressedObservation()
        {
            var wrapped = new Dummy3DSensor();
            wrapped.Shape = new int[] { 1, 1, 3 };
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
    }
}
