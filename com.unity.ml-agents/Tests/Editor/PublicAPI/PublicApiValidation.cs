using System.Collections.Generic;
using MLAgents;
using MLAgents.Policies;
using MLAgents.Sensors;
using NUnit.Framework;
using UnityEngine;

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


        [Test]
        public void CheckSetupAgent()
        {
            var gameObject = new GameObject();

            var behaviorParams = gameObject.AddComponent<BehaviorParameters>();
            behaviorParams.brainParameters.vectorObservationSize = 3;
            behaviorParams.brainParameters.numStackedVectorObservations = 2;
            behaviorParams.brainParameters.vectorActionDescriptions = new[] { "TestActionA", "TestActionB" };
            behaviorParams.brainParameters.vectorActionSize = new[] { 2, 2 };
            behaviorParams.brainParameters.vectorActionSpaceType = SpaceType.Discrete;
            behaviorParams.behaviorName = "TestBehavior";
            behaviorParams.TeamId = 42;
            behaviorParams.useChildSensors = true;
            behaviorParams.behaviorType = BehaviorType.HeuristicOnly;

            var agent = gameObject.AddComponent<Agent>();

            // TODO -  not internal yet
            // var decisionRequester = gameObject.AddComponent<DecisionRequester>();
            // decisionRequester.DecisionPeriod = 2;

            var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
            sensorComponent.sensorName = "ray3d";
            sensorComponent.detectableTags = new List<string> { "Player", "Respawn" };
            sensorComponent.raysPerDirection = 3;

            // ISensor isn't set up yet.
            Assert.IsNull(sensorComponent.raySensor);

            agent.LazyInitialize();

            // Initialization should set up the sensors
            Assert.IsNotNull(sensorComponent.raySensor);

            // Let's change the inference device
            var otherDevice = behaviorParams.inferenceDevice == InferenceDevice.CPU ? InferenceDevice.GPU : InferenceDevice.CPU;
            agent.SetModel(behaviorParams.behaviorName, behaviorParams.model, otherDevice);

            agent.AddReward(1.0f);

            agent.RequestAction();
            agent.RequestDecision();

            Academy.Instance.AutomaticSteppingEnabled = false;
            Academy.Instance.EnvironmentStep();

            var actions = agent.GetAction();
            // default Heuristic implementation should return zero actions.
            Assert.AreEqual(new [] {0.0f, 0.0f}, actions);
        }
    }
}
