using NUnit.Framework;
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

        [Test]
        public void TestStackingMapping()
        {
            // Test grayscale stacked mapping with CameraSensor
            CameraSensor cameraSensor = new CameraSensor(new Camera(), 64, 64,
                true, "grayscaleCamera", SensorCompressionType.PNG);
            ISparseChannelSensor stackedCameraSensor = new StackingSensor(cameraSensor, 2);
            Assert.AreEqual(stackedCameraSensor.GetCompressedChannelMapping(), new[] { 0, 0, 0, 1, 1, 1 });

            // Test RGB stacked mapping with RenderTextureSensor
            RenderTextureSensor renderTextureSensor = new RenderTextureSensor(new RenderTexture(24, 16, 0),
                false, "renderTexture", SensorCompressionType.PNG);
            ISparseChannelSensor stackedRenderTextureSensor = new StackingSensor(renderTextureSensor, 2);
            Assert.AreEqual(stackedRenderTextureSensor.GetCompressedChannelMapping(), new[] { 0, 1, 2, 3, 4, 5 });
        }
    }
}
