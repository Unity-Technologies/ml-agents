#if MLA_INPUT_TESTS
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    public class ButtonInputActionAdaptorTests : InputTestFixture
    {
        ButtonInputActionAdaptor m_Adaptor;
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
                        { ""name"" : ""button"", ""layout"" : ""Button"" }
                    ]
                }";
            InputSystem.RegisterLayout(kLayout);
            m_Device = InputSystem.AddDevice("TestDevice");
            m_Control = (InputControl<float>)m_Device["button"];
            m_Action = new InputAction("action", InputActionType.Button, "/TestDevice/button", null, null, "Button");
            m_Action.Enable();
            m_Adaptor = new ButtonInputActionAdaptor();
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
            Assert.IsTrue(actionSpec.NumDiscreteActions == 1);
            Assert.IsTrue(actionSpec.BranchSizes[0] == 2);
        }

        [Test]
        public void TestQueueEvent()
        {
            var actionBuffers = new ActionBuffers(ActionSegment<float>.Empty, new ActionSegment<int>(new[] { 1 }));
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
            var actionBuffers = new ActionBuffers(ActionSegment<float>.Empty, new ActionSegment<int>(new[] { 1 }));
            var context = new InputActuatorEventContext(1, m_Device);
            using (context.GetEventForFrame(out var eventPtr))
            {
                m_Adaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, new ActionSpec(), actionBuffers);
            }
            InputSystem.Update();
            var buffer = new ActionBuffers(ActionSegment<float>.Empty, new ActionSegment<int>(new[] { 1 }));
            m_Adaptor.WriteToHeuristic(m_Action, buffer);
            Assert.IsTrue(buffer.DiscreteActions[0] == 1);
        }
    }
}
#endif // MLA_INPUT_TESTS
