using System;
using NUnit.Framework;
using UnityEngine;
using MLAgents.Sensors;

namespace MLAgents.Tests
{

    [TestFixture]
    public class CameraSensorComponentTest
    {
        [Test]
        public void TestCameraSensorComponent()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
                    var camera = Camera.main;

                    var agentGameObj = new GameObject("agent");

                    var cameraComponent = agentGameObj.AddComponent<CameraSensorComponent>();
                    cameraComponent.camera = camera;
                    cameraComponent.height = height;
                    cameraComponent.width = width;
                    cameraComponent.grayscale = grayscale;
                    cameraComponent.compression = compression;

                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, cameraComponent.GetObservationShape());
                    Assert.IsTrue(cameraComponent.IsVisual());
                    Assert.IsFalse(cameraComponent.IsVector());

                    var sensor = cameraComponent.CreateSensor();
                    Assert.AreEqual(expectedShape, sensor.GetObservationShape());
                    Assert.AreEqual(typeof(CameraSensor), sensor.GetType());
                }
            }
        }
    }
}
