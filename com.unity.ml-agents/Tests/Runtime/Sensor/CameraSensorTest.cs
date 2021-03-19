using System;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Object = System.Object;

namespace Unity.MLAgents.Tests
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
                    var c = new GameObject();
                    if (ReferenceEquals(null, camera))
                    {
                        camera = c.AddComponent<Camera>();
                    }
                    var sensor = new CameraSensor(camera, width, height, grayscale, "TestCameraSensor", compression);

                    var obsWriter = new ObservationWriter();
                    var obs = sensor.GetObservationProto(obsWriter);

                    Assert.AreEqual((int)compression, (int)obs.CompressionType);
                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, obs.Shape);
                    UnityEngine.Object.DestroyImmediate(c);
                }
            }
        }
    }
}
