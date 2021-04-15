using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace Unity.MLAgents.Tests.Policies
{
    [TestFixture]
    public class HeuristicPolicyTest
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        /// <summary>
        /// Assert that the action buffers are initialized to zero, and then set them to non-zero values.
        /// </summary>
        /// <param name="actionsOut"></param>
        static void CheckAndSetBuffer(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            for (var continuousIndex = 0; continuousIndex < continuousActions.Length; continuousIndex++)
            {
                Assert.AreEqual(continuousActions[continuousIndex], 0.0f);
                continuousActions[continuousIndex] = 1.0f;
            }

            var discreteActions = actionsOut.DiscreteActions;
            for (var discreteIndex = 0; discreteIndex < discreteActions.Length; discreteIndex++)
            {
                Assert.AreEqual(discreteActions[discreteIndex], 0);
                discreteActions[discreteIndex] = 1;
            }
        }


        class ActionClearedAgent : Agent
        {
            public int HeuristicCalls;
            public override void Heuristic(in ActionBuffers actionsOut)
            {
                CheckAndSetBuffer(actionsOut);
                HeuristicCalls++;
            }
        }

        class ActionClearedActuator : IActuator
        {
            public int HeuristicCalls;
            public ActionClearedActuator(ActionSpec actionSpec)
            {
                ActionSpec = actionSpec;
                Name = GetType().Name;
            }

            public void OnActionReceived(ActionBuffers actionBuffers)
            {
            }

            public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
            {
            }

            public void Heuristic(in ActionBuffers actionBuffersOut)
            {
                CheckAndSetBuffer(actionBuffersOut);
                HeuristicCalls++;
            }

            public ActionSpec ActionSpec { get; }
            public string Name { get; }

            public void ResetData()
            {

            }
        }

        class ActionClearedActuatorComponent : ActuatorComponent
        {
            public ActionClearedActuator ActionClearedActuator;
            public ActionClearedActuatorComponent()
            {
                ActionSpec = new ActionSpec(2, new[] { 3, 3 });
            }

            public override IActuator[] CreateActuators()
            {
                ActionClearedActuator = new ActionClearedActuator(ActionSpec);
                return new IActuator[] { ActionClearedActuator };
            }

            public override ActionSpec ActionSpec { get; }
        }

        [Test]
        public void TestActionsCleared()
        {
            var gameObj = new GameObject();
            var agent = gameObj.AddComponent<ActionClearedAgent>();
            var behaviorParameters = agent.GetComponent<BehaviorParameters>();
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(1, new[] { 4 });
            behaviorParameters.BrainParameters.VectorObservationSize = 0;
            behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;

            var actuatorComponent = gameObj.AddComponent<ActionClearedActuatorComponent>();
            agent.LazyInitialize();

            const int k_NumSteps = 5;
            for (var i = 0; i < k_NumSteps; i++)
            {
                agent.RequestDecision();
                Academy.Instance.EnvironmentStep();
            }

            Assert.AreEqual(agent.HeuristicCalls, k_NumSteps);
            Assert.AreEqual(actuatorComponent.ActionClearedActuator.HeuristicCalls, k_NumSteps);
        }
    }
}
