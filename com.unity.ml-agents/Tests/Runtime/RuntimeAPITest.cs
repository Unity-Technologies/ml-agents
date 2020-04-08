#if UNITY_INCLUDE_TESTS
using System.Collections;
using System.Collections.Generic;
using MLAgents;
using MLAgents.Policies;
using MLAgents.Sensors;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Tests
{

    public class PublicApiAgent : Agent
    {
        public int numHeuristicCalls;

        public override float[] Heuristic()
        {
            numHeuristicCalls++;
            return base.Heuristic();
        }
    }// Simple SensorComponent that sets up a StackingSensor
    public class StackingComponent : SensorComponent
    {
        public SensorComponent wrappedComponent;
        public int numStacks;

        public override ISensor CreateSensor()
        {
            var wrappedSensor = wrappedComponent.CreateSensor();
            return new StackingSensor(wrappedSensor, numStacks);
        }

        public override int[] GetObservationShape()
        {
            int[] shape = (int[]) wrappedComponent.GetObservationShape().Clone();
            for (var i = 0; i < shape.Length; i++)
            {
                shape[i] *= numStacks;
            }

            return shape;
        }
    }

    public class RuntimeApiTest
    {
        [SetUp]
        public static void Setup()
        {
            Academy.Instance.AutomaticSteppingEnabled = false;
        }

        [UnityTest]
        public IEnumerator RuntimeApiTestWithEnumeratorPasses()
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


            // Can't actually create an Agent with InferenceOnly and no model, so change back
            behaviorParams.behaviorType = BehaviorType.Default;

            var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
            sensorComponent.sensorName = "ray3d";
            sensorComponent.detectableTags = new List<string> { "Player", "Respawn" };
            sensorComponent.raysPerDirection = 3;

            // Make a StackingSensor that wraps the RayPerceptionSensorComponent3D
            // This isn't necessarily practical, just to ensure that it can be done
            var wrappingSensorComponent = gameObject.AddComponent<StackingComponent>();
            wrappingSensorComponent.wrappedComponent = sensorComponent;
            wrappingSensorComponent.numStacks = 3;

            // ISensor isn't set up yet.
            Assert.IsNull(sensorComponent.raySensor);


            // Make sure we can set the behavior type correctly after the agent is initialized
            // (this creates a new policy).
            behaviorParams.behaviorType = BehaviorType.HeuristicOnly;

            // Agent needs to be added after everything else is setup.
            var agent = gameObject.AddComponent<PublicApiAgent>();

            // DecisionRequester has to be added after Agent.
            var decisionRequester = gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.TakeActionsBetweenDecisions = true;


            // Initialization should set up the sensors
            Assert.IsNotNull(sensorComponent.raySensor);

            // Let's change the inference device
            var otherDevice = behaviorParams.inferenceDevice == InferenceDevice.CPU ? InferenceDevice.GPU : InferenceDevice.CPU;
            agent.SetModel(behaviorParams.behaviorName, behaviorParams.model, otherDevice);

            agent.AddReward(1.0f);

            // skip a frame.
            yield return null;

            Academy.Instance.EnvironmentStep();

            var actions = agent.GetAction();
            // default Heuristic implementation should return zero actions.
            Assert.AreEqual(new[] {0.0f, 0.0f}, actions);
            Assert.AreEqual(1, agent.numHeuristicCalls);

            Academy.Instance.EnvironmentStep();
            Assert.AreEqual(1, agent.numHeuristicCalls);

            Academy.Instance.EnvironmentStep();
            Assert.AreEqual(2, agent.numHeuristicCalls);
        }
    }
}
#endif
