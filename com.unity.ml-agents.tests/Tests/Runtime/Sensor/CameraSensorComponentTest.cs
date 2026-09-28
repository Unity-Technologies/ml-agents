using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Utils.Tests;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class CameraSensorComponentTest
    {
        static readonly BindingFlags k_Flags = BindingFlags.Instance | BindingFlags.NonPublic;

        readonly List<GameObject> m_GameObjects = new List<GameObject>();

        static bool IsDisposed(ISensor sensor)
        {
            return ((CameraSensor)sensor).IsDisposed;
        }

        GameObject NewGameObject(string name)
        {
            var go = new GameObject(name);
            m_GameObjects.Add(go);
            return go;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var go in m_GameObjects)
            {
                if (go != null)
                {
                    UnityEngine.Object.DestroyImmediate(go);
                }
            }
            m_GameObjects.Clear();

            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

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
                    cameraComponent.RuntimeCameraEnable = true;

                    var sensor = cameraComponent.CreateSensors()[0];
                    var expectedShape = new InplaceArray<int>(grayscale ? 1 : 3, height, width);
                    Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);
                    Assert.AreEqual(typeof(CameraSensor), sensor.GetType());

                    var flags = BindingFlags.Instance | BindingFlags.NonPublic;
                    var runtimeCameraEnabled = (bool)typeof(CameraSensorComponent).GetField("m_RuntimeCameraEnable", flags).GetValue(cameraComponent);
                    Assert.True(runtimeCameraEnabled);

                    // Make sure cleaning up the component cleans up the sensor too
                    cameraComponent.Dispose();
                    var cameraComponentSensors = (List<CameraSensor>)typeof(CameraSensorComponent).GetField("m_Sensors", flags).GetValue(cameraComponent);
                    Assert.IsEmpty(cameraComponentSensors);
                    var cameraTexture = (Texture2D)typeof(CameraSensor).GetField("m_Texture", flags).GetValue(sensor);
                    Assert.IsNull(cameraTexture);
                }
            }
        }

        [Test]
        public void CreateTwoSensors()
        {
            var cam = NewGameObject("SensorCam");
            var camera = cam.AddComponent<Camera>();
            var cameraComponent = cam.AddComponent<CameraSensorComponent>();
            cameraComponent.Camera = camera;
            cameraComponent.RuntimeCameraEnable = true;

            var firstSensor = cameraComponent.CreateSensors()[0];
            Assert.IsFalse(IsDisposed(firstSensor), "the first sensor should start out with a texture");

            var secondSensor = cameraComponent.CreateSensors()[0];
            Assert.IsFalse(IsDisposed(secondSensor));

            Assert.IsFalse(IsDisposed(firstSensor), "the second CreateSensors() destroyed the first sensor's Texture2D");

            Assert.DoesNotThrow(() => firstSensor.Update());
        }

        [Test]
        public void CreateTwoAgentsWithSameSensor()
        {
            var parentGameObject = NewGameObject("OuterAgent");
            var childGameObject = NewGameObject("ChildAgent");
            var cam = NewGameObject("SensorCam");
            cam.transform.parent = childGameObject.transform;

            var camera = cam.AddComponent<Camera>();
            var cameraComponent = cam.AddComponent<CameraSensorComponent>();
            cameraComponent.Camera = camera;
            cameraComponent.RuntimeCameraEnable = true;

            var (parentAgent, childAgent) = TestAgent.CreateNestedAgents(parentGameObject, childGameObject);

            var parentSensor = parentAgent.sensors.Find(s => s is CameraSensor);
            var childSensor = childAgent.sensors.Find(s => s is CameraSensor);

            Assert.IsNotNull(parentSensor, "Parent Agent picked up Child's CameraSensorComponent");
            Assert.IsNotNull(childSensor);
            Assert.AreNotSame(parentSensor, childSensor);

            Assert.IsFalse(IsDisposed(parentSensor), "Parent Agent is holding a sensor whose texture was disposed by Child agent");

            Assert.DoesNotThrow(() => parentSensor.Update());
        }

        [Test]
        public void DisableEnableAgent([Values(1, 3)] int observationStacks)
        {
            var parentGameObject = NewGameObject("OuterAgent");
            var childGameObject = NewGameObject("ChildAgent");
            var cam = NewGameObject("SensorCam");
            cam.transform.parent = childGameObject.transform;

            var camera = cam.AddComponent<Camera>();
            var cameraComponent = cam.AddComponent<CameraSensorComponent>();
            cameraComponent.Camera = camera;
            cameraComponent.ObservationStacks = observationStacks;

            var (_, childAgent) = TestAgent.CreateNestedAgents(parentGameObject, childGameObject);

            var componentSensors = (List<CameraSensor>)typeof(CameraSensorComponent).GetField("m_Sensors", k_Flags).GetValue(cameraComponent);
            Assert.AreEqual(2, componentSensors.Count);
            var parentSensor = componentSensors[0];

            for (var i = 0; i < 5; i++)
            {
                // Agent.OnDisable() disposes its sensors; Agent.OnEnable() re-initializes and calls CreateSensors() again.
                childAgent.enabled = false;
                childAgent.enabled = true;
                cameraComponent.CompressionType = SensorCompressionType.None;
                Assert.AreEqual(2, componentSensors.Count, "disposed sensors should be pruned, not retained");
            }

            Assert.IsFalse(componentSensors.Exists(s => s.IsDisposed));
            Assert.Contains(parentSensor, componentSensors, "the parent Agent's sensor should not be disposed");
        }
    }
}
