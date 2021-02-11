using NUnit.Framework;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using Unity.MLAgents.Policies;
using UnityEditor.VersionControl;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    class TestAdaptor : IRLActionInputAdaptor
    {
        public bool m_EventQueued;
        public bool m_WrittenToHeuristic;

        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeContinuous(1);
        }

        public void QueueInputEventForAction(InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            m_EventQueued = true;
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            m_WrittenToHeuristic = true;
        }

        public void Reset()
        {
            m_EventQueued = false;
            m_WrittenToHeuristic = false;
        }
    }

    [TestFixture]
    public class InputActionActuatorTests
    {
        BehaviorParameters m_BehaviorParameters;
        InputActionActuator m_Actuator;
        TestAdaptor m_Adaptor;

        [SetUp]
        public void Setup()
        {

            var go = new GameObject();
            m_BehaviorParameters = go.AddComponent<BehaviorParameters>();
            var action = new InputAction("action");
            m_Adaptor = new TestAdaptor();
            m_Actuator = new InputActionActuator(m_BehaviorParameters, action, m_Adaptor);

        }

        [Test]
        public void TestIsInHeuristicMode()
        {
            m_BehaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
            Assert.IsTrue(m_Actuator.IsInHeuristicMode());

            m_BehaviorParameters.BehaviorType = BehaviorType.Default;
            Assert.IsTrue(m_Actuator.IsInHeuristicMode());

            m_BehaviorParameters.Model = ScriptableObject.CreateInstance<NNModel>();
            Assert.IsFalse(m_Actuator.IsInHeuristicMode());
        }

        [Test]
        public void TestOnActionReceived()
        {
            m_BehaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
            m_Actuator.OnActionReceived(new ActionBuffers());
            m_Actuator.Heuristic(new ActionBuffers());
            Assert.IsFalse(m_Adaptor.m_EventQueued);
            Assert.IsTrue(m_Adaptor.m_WrittenToHeuristic);
            m_Adaptor.Reset();

            m_BehaviorParameters.BehaviorType = BehaviorType.Default;
            m_Actuator.OnActionReceived(new ActionBuffers());
            Assert.IsFalse(m_Adaptor.m_EventQueued);
            m_Adaptor.Reset();

            m_BehaviorParameters.Model = ScriptableObject.CreateInstance<NNModel>();
            m_Actuator.OnActionReceived(new ActionBuffers());
            Assert.IsTrue(m_Adaptor.m_EventQueued);
            m_Adaptor.Reset();

            Assert.AreEqual(m_Actuator.Name, "InputActionActuator-action");
            m_Actuator.ResetData();
            m_Actuator.WriteDiscreteActionMask(null);
        }

    }
}
