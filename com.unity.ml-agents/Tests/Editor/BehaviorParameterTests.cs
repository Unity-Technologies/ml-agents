using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class BehaviorParameterTests
    {
        static void DummyHeuristic(float[] actionsOut)
        {
            // No-op
        }

        [Test]
        public void TestNoModelInferenceOnlyThrows()
        {
            var gameObj = new GameObject();
            var bp = gameObj.AddComponent<BehaviorParameters>();
            bp.BehaviorType = BehaviorType.InferenceOnly;

            Assert.Throws<UnityAgentsException>(() =>
            {
                bp.GeneratePolicy(DummyHeuristic);
            });
        }
    }
}
