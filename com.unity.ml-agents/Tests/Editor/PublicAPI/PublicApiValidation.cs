using System.Collections.Generic;
using MLAgents;
using MLAgents.Policies;
using MLAgents.Sensors;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace MLAgentsExamples
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
            sensorComponent.camera = Camera.main;
            sensorComponent.sensorName = "camera1";
            sensorComponent.width = width;
            sensorComponent.height = height;
            sensorComponent.grayscale = true;

            // Make sure the sets actually applied
            Assert.AreEqual("camera1", sensorComponent.sensorName);
            Assert.AreEqual(width, sensorComponent.width);
            Assert.AreEqual(height, sensorComponent.height);
            Assert.IsTrue(sensorComponent.grayscale);
        }

        [Test]
        public void CheckSetupRenderTextureSensorComponent()
        {
            var gameObject = new GameObject();

            var sensorComponent = gameObject.AddComponent<RenderTextureSensorComponent>();
            var width = 24;
            var height = 16;
            var texture = new RenderTexture(width, height, 0);
            sensorComponent.renderTexture = texture;
            sensorComponent.sensorName = "rtx1";
            sensorComponent.grayscale = true;

            // Make sure the sets actually applied
            Assert.AreEqual("rtx1", sensorComponent.sensorName);
            Assert.IsTrue(sensorComponent.grayscale);
        }

        [Test]
        public void CheckSetupRayPerceptionSensorComponent()
        {
            var gameObject = new GameObject();

            var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
            sensorComponent.sensorName = "ray3d";
            sensorComponent.detectableTags = new List<string> { "Player", "Respawn" };
            sensorComponent.raysPerDirection = 3;
            sensorComponent.maxRayDegrees = 30;
            sensorComponent.sphereCastRadius = .1f;
            sensorComponent.rayLayerMask = 0;
            sensorComponent.observationStacks = 2;

            sensorComponent.CreateSensor();
        }
    }
}
