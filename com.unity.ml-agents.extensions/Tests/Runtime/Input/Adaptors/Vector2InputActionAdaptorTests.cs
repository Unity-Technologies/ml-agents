#if MLA_INPUT_TESTS
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    public class Vector2InputActionAdaptorTests : InputTestFixture
    {
        Vector2InputActionAdaptor m_Adaptor;
        InputDevice m_Device;
        InputControl<Vector2> m_Control;
        InputAction m_Action;

        public override void Setup()
        {
            base.Setup();
            const string kLayout = @"
                {
                    ""name"" : ""TestDevice"",
                    ""extend"" : ""HID"",
                    ""controls"" : [
                        { ""name"" : ""button"", ""layout"" : ""Vector2"" }
                    ]
                }";
            InputSystem.RegisterLayout(kLayout);
            m_Device = InputSystem.AddDevice("TestDevice");
            m_Control = (InputControl<Vector2>)m_Device["button"];
            m_Action = new InputAction("action", InputActionType.Value, "/TestDevice/button", null, null, "Vector2");
            m_Action.Enable();
            m_Adaptor = new Vector2InputActionAdaptor();
        }

        public override void TearDown()
        {
            base.TearDown();
            m_Adaptor = null;
        }

        [Test]
        public void TestGenerateActionSpec()
        {
            var actionSpec = m_Adaptor.GetActionSpecForInputAction(new InputAction());
            Assert.IsTrue(actionSpec.NumContinuousActions == 2);
        }

        [Test]
        public void TestQueueEvent()
        {
            var actionBuffers = new ActionBuffers(new ActionSegment<float>(new[] { 0f, 1f }), ActionSegment<int>.Empty);
            var context = new InputActuatorEventContext(1, m_Device);
            using (context.GetEventForFrame(out var eventPtr))
            {
                m_Adaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, new ActionSpec(), actionBuffers);
            }
            InputSystem.Update();
            var val = m_Action.ReadValue<Vector2>();
            Assert.IsTrue(Mathf.Approximately(0f, val.x));
            Assert.IsTrue(Mathf.Approximately(1f, val.y));
        }

        [Test]
        public void TestWriteToHeuristic()
        {
            var actionBuffers = new ActionBuffers(new ActionSegment<float>(new[] { 0f, 1f }), ActionSegment<int>.Empty);
            var context = new InputActuatorEventContext(1, m_Device);
            using (context.GetEventForFrame(out var eventPtr))
            {
                m_Adaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, new ActionSpec(), actionBuffers);
            }
            InputSystem.Update();
            var buffer = new ActionBuffers(new ActionSegment<float>(new float[2]), ActionSegment<int>.Empty);
            m_Adaptor.WriteToHeuristic(m_Action, buffer);
            Assert.IsTrue(Mathf.Approximately(buffer.ContinuousActions[0], 0f));
            Assert.IsTrue(Mathf.Approximately(buffer.ContinuousActions[1], 1f));
        }
    }
}
#endif // MLA_INPUT_TESTS
