using NUnit.Framework;
using System;
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
        public void TestStacking()
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
        public void TestStackingReset()
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

        class DummySensor : ISparseChannelSensor
        {
            public SensorCompressionType CompressionType = SensorCompressionType.PNG;
            public int[] Mapping;
            public int[] Shape = new int[] { 8, 8, 3 };

            internal DummySensor()
            {
            }

            public int[] GetObservationShape()
            {
                return Shape;
            }

            public int Write(ObservationWriter writer)
            {
                return 0;
            }

            public byte[] GetCompressedObservation()
            {
                var obs = new int[] { 1, 2, 3 };
                byte[] bytes = Array.ConvertAll(obs, (z) => (byte)z);
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

            // Test mapping with dummy layers that should be dropped
            var dummySensor = new DummySensor();
            dummySensor.Shape = new int[] { 2, 2, 4 };
            dummySensor.Mapping = new int[] { 0, 1, 2, 3, -1, -1 };
            var stackedDummySensor = new StackingSensor(dummySensor, 2);
            Assert.AreEqual(stackedDummySensor.GetCompressedChannelMapping(), new[] { 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1 });
        }

        [Test]
        public void TestStackedGetCompressedObservation()
        {
            var dummySensor = new DummySensor();
            var stackedDummySensor = new StackingSensor(dummySensor, 3);
            // Call three times to fill the buffer
            stackedDummySensor.GetCompressedObservation();
            stackedDummySensor.Update();
            stackedDummySensor.GetCompressedObservation();
            stackedDummySensor.Update();
            var compressedObs = stackedDummySensor.GetCompressedObservation();

            int[] decompressed = Array.ConvertAll(compressedObs, c => (int)c);
            var expectedObservation = new int[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            Assert.AreEqual(decompressed, expectedObservation);
        }
    }
}
