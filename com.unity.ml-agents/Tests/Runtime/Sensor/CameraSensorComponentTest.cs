using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
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
                    cameraComponent.Camera = camera;
                    cameraComponent.Height = height;
                    cameraComponent.Width = width;
                    cameraComponent.Grayscale = grayscale;
                    cameraComponent.CompressionType = compression;

                    var sensor = cameraComponent.CreateSensors()[0];
                    var expectedShape = new InplaceArray<int>(height, width, grayscale ? 1 : 3);
                    Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);
                    Assert.AreEqual(typeof(CameraSensor), sensor.GetType());

                    // Make sure cleaning up the component cleans up the sensor too
                    cameraComponent.Dispose();
                    var flags = BindingFlags.Instance | BindingFlags.NonPublic;
                    var cameraComponentSensor = (CameraSensor)typeof(CameraSensorComponent).GetField("m_Sensor", flags).GetValue(cameraComponent);
                    Assert.IsNull(cameraComponentSensor);
                    var cameraTexture = (Texture2D)typeof(CameraSensor).GetField("m_Texture", flags).GetValue(sensor);
                    Assert.IsNull(cameraTexture);
                }
            }
        }
    }
}
