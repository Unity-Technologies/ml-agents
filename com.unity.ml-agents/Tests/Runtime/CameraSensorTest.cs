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

        [UnityTest]
        public IEnumerator TestCameraSensor()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
//                    var camera = new Camera();
//                    var sensor = new CameraSensor(camera, width, height, grayscale, "TestCameraSensor", compression);
//
//                    var writeAdapter = new WriteAdapter();
//                    var obs = sensor.GetObservationProto(writeAdapter);

                    //yield return new WaitForFixedUpdate();
                    var agentGameObj = new GameObject("agent");
                    var agent = agentGameObj.AddComponent<CameraTestAgent>();

                    var cameraComponent = agentGameObj.AddComponent<CameraSensorComponent>();
                    cameraComponent.camera = new Camera();
                    cameraComponent.height = height;
                    cameraComponent.width = width;
                    cameraComponent.grayscale = grayscale;
                    cameraComponent.compression = compression;
                    for (var i=0; i<5; i++)
                    {
                        // TODO make sure we actually do something in the step - Sensors aren't stepped for Heuristic
                        yield return new WaitForFixedUpdate();
                    }



                }
            }
        }
    }
}
