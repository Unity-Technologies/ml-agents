using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

using Unity.MLAgents.Tests;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class RigidBodySensorTests
    {
        [Test]
        public void TestNullRootBody()
        {
            var gameObj = new GameObject();

            var sensorComponent = gameObj.AddComponent<RigidBodySensorComponent>();
            var sensor = sensorComponent.CreateSensor();
            SensorTestHelper.CompareObservation(sensor, new float[0]);
        }
    }
}
