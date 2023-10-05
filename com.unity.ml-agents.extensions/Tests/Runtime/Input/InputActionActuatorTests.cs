#if MLA_INPUT_TESTS
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using Unity.MLAgents.Policies;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    class TestAdaptor : IRLActionInputAdaptor
    {
        public bool eventWritten;
        public bool writtenToHeuristic;

        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeContinuous(1);
        }

        public void WriteToInputEventForAction(InputEventPtr eventPtr, InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            eventWritten = true;
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            writtenToHeuristic = true;
        }

        public void Reset()
        {
            eventWritten = false;
            writtenToHeuristic = false;
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
            m_Actuator = new InputActionActuator(null, m_BehaviorParameters, action, m_Adaptor, new InputActuatorEventContext(1, InputSystem.AddDevice<Gamepad>()));
        }

        [Test]
        public void TestOnActionReceived()
        {
            m_BehaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
            m_Actuator.OnActionReceived(new ActionBuffers());
            m_Actuator.Heuristic(new ActionBuffers());
            Assert.IsFalse(m_Adaptor.eventWritten);
            Assert.IsTrue(m_Adaptor.writtenToHeuristic);
            m_Adaptor.Reset();

            m_BehaviorParameters.BehaviorType = BehaviorType.Default;
            m_Actuator.OnActionReceived(new ActionBuffers());
            Assert.IsFalse(m_Adaptor.eventWritten);
            m_Adaptor.Reset();

            m_BehaviorParameters.Model = ScriptableObject.CreateInstance<ModelAsset>();
            m_Actuator.OnActionReceived(new ActionBuffers());
            Assert.IsTrue(m_Adaptor.eventWritten);
            m_Adaptor.Reset();

            Assert.AreEqual(m_Actuator.Name, "InputActionActuator-action");
            m_Actuator.ResetData();
            m_Actuator.WriteDiscreteActionMask(null);
        }
    }
}
#endif // MLA_INPUT_TESTS
