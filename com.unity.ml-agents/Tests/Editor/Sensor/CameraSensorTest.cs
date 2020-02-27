 using System;
 using System.Collections;
 using NUnit.Framework;
 #if UNITY_EDITOR
 using UnityEditor.SceneManagement;
 using UnityEngine.TestTools;
 #endif
 using UnityEngine;
 using UnityEngine.SceneManagement;
 using MLAgents;
 using MLAgents.Sensors;

namespace MLAgents.Tests
{

    [TestFixture]
    public class CameraSensorTest
    {
        private class CameraTestAgent : Agent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
                base.CollectObservations(sensor);
            }

            public override float[] Heuristic()
            {
                return new [] {1.0f};
            }

        }

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
                    var sensor = new CameraSensor(Camera.main, width, height, grayscale, "TestCameraSensor", compression);

                    var writeAdapter = new WriteAdapter();
                    var obs = sensor.GetObservationProto(writeAdapter);

                    Assert.AreEqual((int) compression, (int) obs.CompressionType);
                    var expectedShape = new[] { height, width, grayscale ? 1 : 3 };
                    Assert.AreEqual(expectedShape, obs.Shape);


//
//                    var agentGameObj = new GameObject("agent");
//                    var agent = agentGameObj.AddComponent<CameraTestAgent>();
//
//                    var cameraComponent = agentGameObj.AddComponent<CameraSensorComponent>();
//                    cameraComponent.camera = new Camera();
//                    cameraComponent.height = height;
//                    cameraComponent.width = width;
//                    cameraComponent.grayscale = grayscale;
//                    cameraComponent.compression = compression;




                }
            }
        }
    }
}
