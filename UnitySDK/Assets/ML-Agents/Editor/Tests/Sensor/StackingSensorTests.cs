using NUnit.Framework;
using UnityEngine;
using MLAgents.Sensor;

namespace MLAgents.Tests
{
    public class StackingSensorTest
    {
        [Test]
        public void TestCtor()
        {
            ISensor wrapped = new VectorSensor(4);
            ISensor sensor = new StackingSensor(wrapped, 4);
            Assert.AreEqual("StackingSensor_size4_VectorSensor_size4", sensor.GetName());
            Assert.AreEqual(sensor.GetFloatObservationShape(), new [] {16});
        }

        [Test]
        public void TestStacking()
        {
            VectorSensor wrapped = new VectorSensor(2);
            ISensor sensor = new StackingSensor(wrapped, 3);

            wrapped.AddObservation(new [] {1f, 2f});
            SensorTestHelper.CompareObservation(sensor, new [] {0f, 0f, 0f, 0f, 1f, 2f});

            wrapped.AddObservation(new [] {3f, 4f});
            SensorTestHelper.CompareObservation(sensor, new [] {0f, 0f, 1f, 2f, 3f, 4f});

            wrapped.AddObservation(new [] {5f, 6f});
            SensorTestHelper.CompareObservation(sensor, new [] {1f, 2f, 3f, 4f, 5f, 6f});

            wrapped.AddObservation(new [] {7f, 8f});
            SensorTestHelper.CompareObservation(sensor, new [] {3f, 4f, 5f, 6f, 7f, 8f});

            wrapped.AddObservation(new [] {9f, 10f});
            SensorTestHelper.CompareObservation(sensor, new [] {5f, 6f, 7f, 8f, 9f, 10f});
        }


    }
}
