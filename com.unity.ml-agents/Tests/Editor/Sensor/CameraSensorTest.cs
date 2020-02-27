 using System;
 using NUnit.Framework;
 using UnityEngine;
 using MLAgents.Sensors;

namespace MLAgents.Tests
{

    [TestFixture]
    public class CameraSensorTest
    {
        [Test]
        public void TestCameraSensor()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
                    var camera = Camera.main;
                    var sensor = new CameraSensor(camera, width, height, grayscale, "TestCameraSensor", compression);

                    var writeAdapter = new WriteAdapter();
                    var obs = sensor.GetObservationProto(writeAdapter);

                    Assert.AreEqual((int) compression, (int) obs.CompressionType);
                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, obs.Shape);
                }
            }
        }
    }
}
