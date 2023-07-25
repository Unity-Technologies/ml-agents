using Unity.MLAgents.Sensors;
using NUnit.Framework;
using Unity.MLAgents;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// The purpose of these tests is to make sure that we can do basic operations like creating
    /// an Agent and adding components from code without requiring access to internal methods.
    /// The tests aren't intended to add extra test coverage (although they might) and might
    /// not check any conditions.
    /// </summary>
    [TestFixture]
    public class PublicApiValidation
    {
        [Test]
        public void CheckSetupCameraSensorComponent()
        {
            var gameObject = new GameObject();
            var width = 24;
            var height = 16;

            var sensorComponent = gameObject.AddComponent<CameraSensorComponent>();
            sensorComponent.Camera = Camera.main;
            sensorComponent.SensorName = "camera1";
            sensorComponent.Width = width;
            sensorComponent.Height = height;
            sensorComponent.Grayscale = true;

            // Make sure the sets actually applied
            Assert.AreEqual("camera1", sensorComponent.SensorName);
            Assert.AreEqual(width, sensorComponent.Width);
            Assert.AreEqual(height, sensorComponent.Height);
            Assert.IsTrue(sensorComponent.Grayscale);
        }

        [Test]
        public void CheckSetupRenderTextureSensorComponent()
        {
            var gameObject = new GameObject();

            var sensorComponent = gameObject.AddComponent<RenderTextureSensorComponent>();
            var width = 24;
            var height = 16;
            var texture = new RenderTexture(width, height, 0);
            sensorComponent.RenderTexture = texture;
            sensorComponent.SensorName = "rtx1";
            sensorComponent.Grayscale = true;

            // Make sure the sets actually applied
            Assert.AreEqual("rtx1", sensorComponent.SensorName);
            Assert.IsTrue(sensorComponent.Grayscale);
        }

#if MLA_UNITY_PHYSICS_MODULE
        [Test]
        public void CheckSetupRayPerceptionSensorComponent()
        {
            var gameObject = new GameObject();

            var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
            sensorComponent.SensorName = "ray3d";
            sensorComponent.DetectableTags = new List<string> { "Player", "Respawn" };
            sensorComponent.RaysPerDirection = 3;
            sensorComponent.MaxRayDegrees = 30;
            sensorComponent.SphereCastRadius = .1f;
            sensorComponent.RayLayerMask = 0;
            sensorComponent.ObservationStacks = 2;

            sensorComponent.CreateSensors();

            var sensor = sensorComponent.RaySensor;
            sensor.Update();
            var outputs = sensor.RayPerceptionOutput;
            Assert.AreEqual(outputs.RayOutputs.Length, 2 * sensorComponent.RaysPerDirection + 1);
        }

#endif

        /// <summary>
        /// Make sure we can inherit from DecisionRequester and override some logic.
        /// </summary>
        class CustomDecisionRequester : DecisionRequester
        {
            /// <summary>
            /// Example logic. If the killswitch flag is set, the Agent never requests a decision.
            /// </summary>
            public bool KillswitchEnabled;

            public CustomDecisionRequester()
            {
            }

            protected override bool ShouldRequestDecision(DecisionRequestContext context)
            {
                return !KillswitchEnabled && base.ShouldRequestDecision(context);
            }
        }
    }
}
