using System;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class RenderTextureSensorComponentTest
    {
        [Test]
        public void TestRenderTextureSensorComponent()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
                    var texture = new RenderTexture(width, height, 0);

                    var agentGameObj = new GameObject("agent");

                    var renderTexComponent = agentGameObj.AddComponent<RenderTextureSensorComponent>();
                    renderTexComponent.RenderTexture = texture;
                    renderTexComponent.Grayscale = grayscale;
                    renderTexComponent.CompressionType = compression;

                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, renderTexComponent.GetObservationShape());
                    Assert.IsTrue(renderTexComponent.IsVisual());
                    Assert.IsFalse(renderTexComponent.IsVector());

                    var sensor = renderTexComponent.CreateSensor();
                    Assert.AreEqual(expectedShape, sensor.GetObservationShape());
                    Assert.AreEqual(typeof(RenderTextureSensor), sensor.GetType());
                }
            }
        }
    }
}
