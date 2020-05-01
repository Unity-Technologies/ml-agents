using System;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class RenderTextureSensorTests
    {
        [Test]
        public void TestRenderTextureSensor()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
                    var texture = new RenderTexture(width, height, 0);
                    var sensor = new RenderTextureSensor(texture, grayscale, "TestCameraSensor", compression);

                    var obsWriter = new ObservationWriter();
                    var obs = sensor.GetObservationProto(obsWriter);

                    Assert.AreEqual((int)compression, (int)obs.CompressionType);
                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, obs.Shape);
                }
            }
        }
    }
}
