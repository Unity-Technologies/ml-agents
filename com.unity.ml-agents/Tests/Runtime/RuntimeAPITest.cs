using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.TestTools;

namespace Tests
{
    public class PublicApiAgent : Agent
    {
        public int numHeuristicCalls;

        [Observable]
        public float ObservableFloat;

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            numHeuristicCalls++;
            base.Heuristic(actionsOut);
        }
    }

    // Simple SensorComponent that sets up a StackingSensor
    public class StackingComponent : SensorComponent
    {
        public SensorComponent wrappedComponent;
        public int numStacks;

        public override ISensor[] CreateSensors()
        {
            var wrappedSensors = wrappedComponent.CreateSensors();
            var sensorsOut = new ISensor[wrappedSensors.Length];
            for (var i = 0; i < wrappedSensors.Length; i++)
            {
                sensorsOut[i] = new StackingSensor(wrappedSensors[i], numStacks);
            }

            return sensorsOut;
        }
    }

    public class RuntimeApiTest
    {
        [SetUp]
        public static void Setup()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
            Academy.Instance.AutomaticSteppingEnabled = false;
        }

        [UnityTest]
        public IEnumerator RuntimeApiTestWithEnumeratorPasses()
        {
            Academy.Instance.InferenceSeed = 1337;
            var gameObject = new GameObject();

            var behaviorParams = gameObject.AddComponent<BehaviorParameters>();
            behaviorParams.BrainParameters.VectorObservationSize = 3;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 2;
            behaviorParams.BrainParameters.VectorActionDescriptions = new[] { "Continuous1", "TestActionA", "TestActionB" };
            behaviorParams.BrainParameters.ActionSpec = new ActionSpec(1, new[] { 2, 2 });
            behaviorParams.BehaviorName = "TestBehavior";
            behaviorParams.TeamId = 42;
            behaviorParams.UseChildSensors = true;
            behaviorParams.ObservableAttributeHandling = ObservableAttributeOptions.ExamineAll;


            // Can't actually create an Agent with InferenceOnly and no model, so change back
            behaviorParams.BehaviorType = BehaviorType.Default;
#if MLA_UNITY_PHYSICS_MODULE
            var sensorComponent = gameObject.AddComponent<RayPerceptionSensorComponent3D>();
            sensorComponent.SensorName = "ray3d";
            sensorComponent.DetectableTags = new List<string> { "Player", "Respawn" };
            sensorComponent.RaysPerDirection = 3;

            // Make a StackingSensor that wraps the RayPerceptionSensorComponent3D
            // This isn't necessarily practical, just to ensure that it can be done
            var wrappingSensorComponent = gameObject.AddComponent<StackingComponent>();
            wrappingSensorComponent.wrappedComponent = sensorComponent;
            wrappingSensorComponent.numStacks = 3;

            // ISensor isn't set up yet.
            Assert.IsNull(sensorComponent.RaySensor);
#endif


            // Make sure we can set the behavior type correctly after the agent is initialized
            // (this creates a new policy).
            behaviorParams.BehaviorType = BehaviorType.HeuristicOnly;

            // Agent needs to be added after everything else is setup.
            var agent = gameObject.AddComponent<PublicApiAgent>();

            // DecisionRequester has to be added after Agent.
            var decisionRequester = gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.TakeActionsBetweenDecisions = true;

#if MLA_UNITY_PHYSICS_MODULE
            // Initialization should set up the sensors
            Assert.IsNotNull(sensorComponent.RaySensor);
#endif
            // Let's change the inference device
            var otherDevice = behaviorParams.InferenceDevice == InferenceDevice.CPU ? InferenceDevice.GPU : InferenceDevice.CPU;
            agent.SetModel(behaviorParams.BehaviorName, behaviorParams.Model, otherDevice);

            agent.AddReward(1.0f);

            // skip a frame.
            yield return null;

            Academy.Instance.EnvironmentStep();

            var actions = agent.GetStoredActionBuffers().DiscreteActions;
            // default Heuristic implementation should return zero actions.
            Assert.AreEqual(new ActionSegment<int>(new[] { 0, 0 }), actions);
            Assert.AreEqual(1, agent.numHeuristicCalls);

            Academy.Instance.EnvironmentStep();
            Assert.AreEqual(1, agent.numHeuristicCalls);

            Academy.Instance.EnvironmentStep();
            Assert.AreEqual(2, agent.numHeuristicCalls);
        }
    }
}
