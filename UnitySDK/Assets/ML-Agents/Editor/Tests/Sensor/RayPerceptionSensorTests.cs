using NUnit.Framework;
using UnityEngine;
using MLAgents.Sensor;

namespace MLAgents.Tests
{
    public class RayPerceptionSensorTests
    {
        [Test]
        public void TestGetRayAngles()
        {
            var angles = RayPerceptionSensorComponentBase.GetRayAngles(3, 90f);
            var expectedAngles = new [] { 90f, 60f, 120f, 30f, 150f, 0f, 180f };
            Assert.AreEqual(expectedAngles.Length, angles.Length);
            for (var i = 0; i < angles.Length; i++)
            {
                Assert.AreEqual(expectedAngles[i], angles[i], .01);
            }
        }
    }
}
