using NUnit.Framework;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine.TestTools;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class BehaviorParameterTests : IHeuristicProvider
    {
        const string k_continuousONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action_v1_0.onnx";
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
                bp.GeneratePolicy(actionSpec, new ActuatorManager());
            });
        }

        [Test]
        public void TestIsInHeuristicMode()
        {
            var gameObj = new GameObject();
            var bp = gameObj.AddComponent<BehaviorParameters>();
            bp.Model = null;
            gameObj.AddComponent<Agent>();
            bp.BehaviorType = BehaviorType.HeuristicOnly;
            Assert.IsTrue(bp.IsInHeuristicMode());

            bp.BehaviorType = BehaviorType.Default;
            Assert.IsTrue(bp.IsInHeuristicMode());

            bp.Model = ScriptableObject.CreateInstance<NNModel>();
            Assert.IsFalse(bp.IsInHeuristicMode());
        }

        [Test]
        public void TestPolicyUpdateEventFired()
        {
            var gameObj = new GameObject();
            var bp = gameObj.AddComponent<BehaviorParameters>();
            gameObj.AddComponent<Agent>().LazyInitialize();
            bp.OnPolicyUpdated += delegate (bool isInHeuristicMode) { Debug.Log($"OnPolicyChanged:{isInHeuristicMode}"); };
            bp.BehaviorType = BehaviorType.HeuristicOnly;
            LogAssert.Expect(LogType.Log, $"OnPolicyChanged:{true}");

            bp.BehaviorType = BehaviorType.Default;
            LogAssert.Expect(LogType.Log, $"OnPolicyChanged:{true}");

            Assert.Throws<UnityAgentsException>(() =>
            {
                bp.BehaviorType = BehaviorType.InferenceOnly;
            });

            bp.Model = AssetDatabase.LoadAssetAtPath<NNModel>(k_continuousONNXPath);
            LogAssert.Expect(LogType.Log, $"OnPolicyChanged:{false}");

            bp.BehaviorType = BehaviorType.HeuristicOnly;
            LogAssert.Expect(LogType.Log, $"OnPolicyChanged:{true}");
        }
    }
}
