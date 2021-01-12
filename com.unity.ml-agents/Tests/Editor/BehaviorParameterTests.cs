using NUnit.Framework;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class BehaviorParameterTests : IHeuristic
    {
        public void Heuristic(in ActionBuffers actionsOut)
        {
            // No-op
        }

        [Test]
        public void TestNoModelInferenceOnlyThrows()
        {
            var gameObj = new GameObject();
            var bp = gameObj.AddComponent<BehaviorParameters>();
            bp.BehaviorType = BehaviorType.InferenceOnly;
            var actionSpec = new ActionSpec();

            Assert.Throws<UnityAgentsException>(() =>
            {
                bp.GeneratePolicy(actionSpec, this);
            });
        }
    }
}
