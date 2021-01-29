using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.Barracuda;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.Policies;
using UnityEditor;

namespace Unity.MLAgents.Tests.Analytics
{
    [TestFixture]
    public class TrainingAnalyticsTests
    {
        [TestCase("foo?team=42", ExpectedResult = "foo")]
        [TestCase("foo", ExpectedResult = "foo")]
        [TestCase("foo?bar?team=1337", ExpectedResult = "foo?bar")]
        public string TestParseBehaviorName(string fullyQualifiedBehaviorName)
        {
            return TrainingAnalytics.ParseBehaviorName(fullyQualifiedBehaviorName);
        }

        [Test]
        public void TestRemotePolicy()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }

            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                var brainParameters = new BrainParameters();
                brainParameters.VectorObservationSize = 8;
                brainParameters.VectorActionSize = new [] { 2 };
                brainParameters.NumStackedVectorObservations = 1;
                brainParameters.VectorActionSpaceType = SpaceType.Continuous;

                var policy = new RemotePolicy(brainParameters, "TestBehavior?team=42");
                policy.RequestDecision(new AgentInfo(), new List<ISensor>());
            }

            Academy.Instance.Dispose();
        }

        [Test]
        public void TestBuiltInSensorType()
        {
            // Unknown
            {
                var sensor = new TestSensor("test");
                Assert.AreEqual(sensor.GetBuiltInSensorType(), BuiltInSensorType.Unknown);

                var stackingSensor = new StackingSensor(sensor, 2);
                Assert.AreEqual(BuiltInSensorType.Unknown, stackingSensor.GetBuiltInSensorType());
            }

            // Vector
            {
                var sensor = new VectorSensor(6);
                Assert.AreEqual(BuiltInSensorType.VectorSensor, sensor.GetBuiltInSensorType());

                var stackingSensor = new StackingSensor(sensor, 2);
                Assert.AreEqual(BuiltInSensorType.VectorSensor, stackingSensor.GetBuiltInSensorType());
            }

            var gameObject = new GameObject();

            // Ray
            {
                var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
                sensorComponent.DetectableTags = new List<string>();
                var sensor = sensorComponent.CreateSensor();
                Assert.AreEqual(BuiltInSensorType.RayPerceptionSensor, sensor.GetBuiltInSensorType());

                var stackingSensor = new StackingSensor(sensor, 2);
                Assert.AreEqual(BuiltInSensorType.RayPerceptionSensor, stackingSensor.GetBuiltInSensorType());
            }

            // Camera
            {
                var sensorComponent = gameObject.AddComponent<CameraSensorComponent>();
                var sensor = sensorComponent.CreateSensor();
                Assert.AreEqual(BuiltInSensorType.CameraSensor, sensor.GetBuiltInSensorType());
            }

            // RenderTexture
            {
                var sensorComponent = gameObject.AddComponent<RenderTextureSensorComponent>();
                var sensor = sensorComponent.CreateSensor();
                Assert.AreEqual(BuiltInSensorType.RenderTextureSensor, sensor.GetBuiltInSensorType());
            }

        }
    }
}
