#if MLA_INPUT_TESTS
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    public class FloatInputActionAdaptorTests : InputTestFixture
    {
        FloatInputActionAdaptor m_Adaptor;
        InputDevice m_Device;
        InputControl<float> m_Control;
        InputAction m_Action;

        public override void Setup()
        {
            base.Setup();
            const string kLayout = @"
                {
                    ""name"" : ""TestDevice"",
                    ""extend"" : ""HID"",
                    ""controls"" : [
                        { ""name"" : ""button"", ""layout"" : ""Axis"" }
                    ]
                }";
            InputSystem.RegisterLayout(kLayout);
            m_Device = InputSystem.AddDevice("TestDevice");
            m_Control = (InputControl<float>)m_Device["button"];
            m_Action = new InputAction("action", InputActionType.Value, "/TestDevice/button", null, null, "Axis");
            m_Action.Enable();
            m_Adaptor = new FloatInputActionAdaptor();
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
            Assert.IsTrue(actionSpec.NumContinuousActions == 1);
        }

        [Test]
        public void TestQueueEvent()
        {
            var actionBuffers = new ActionBuffers(new ActionSegment<float>(new[] { 1f }), ActionSegment<int>.Empty);
            var context = new InputActuatorEventContext(1, m_Device);
            using (context.GetEventForFrame(out var eventPtr))
            {
                m_Adaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, new ActionSpec(), actionBuffers);
            }
            InputSystem.Update();
            var val = m_Action.ReadValue<float>();
            Assert.IsTrue(Mathf.Approximately(1f, val));
        }

        [Test]
        public void TestWriteToHeuristic()
        {
            var actionBuffers = new ActionBuffers(new ActionSegment<float>(new[] { 1f }), ActionSegment<int>.Empty);
            var context = new InputActuatorEventContext(1, m_Device);
            using (context.GetEventForFrame(out var eventPtr))
            {
                m_Adaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, new ActionSpec(), actionBuffers);
            }
            InputSystem.Update();
            var buffer = new ActionBuffers(new ActionSegment<float>(new[] { 1f }), ActionSegment<int>.Empty);
            m_Adaptor.WriteToHeuristic(m_Action, buffer);
            Assert.IsTrue(Mathf.Approximately(1f, buffer.ContinuousActions[0]));
        }
    }
}
#endif // MLA_INPUT_TESTS
