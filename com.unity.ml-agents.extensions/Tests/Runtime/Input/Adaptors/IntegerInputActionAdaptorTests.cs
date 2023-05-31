#if MLA_INPUT_TESTS
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    public class IntegerInputActionAdaptorTests : InputTestFixture
    {
        IntegerInputActionAdaptor m_Adaptor;
        InputDevice m_Device;
        InputControl<int> m_Control;
        InputAction m_Action;

        public override void Setup()
        {
            base.Setup();
            const string kLayout = @"
                {
                    ""name"" : ""TestDevice"",
                    ""extend"" : ""HID"",
                    ""controls"" : [
                        { ""name"" : ""button"", ""layout"" : ""integer"" }
                    ]
                }";
            InputSystem.RegisterLayout(kLayout);
            m_Device = InputSystem.AddDevice("TestDevice");
            m_Control = (InputControl<int>)m_Device["button"];
            m_Action = new InputAction("action", InputActionType.Value, "/TestDevice/button", null, null, "int");
            m_Action.Enable();
            m_Adaptor = new IntegerInputActionAdaptor();
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
            Assert.IsTrue(actionSpec.SumOfDiscreteBranchSizes == 2);
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
            var val = m_Action.ReadValue<int>();
            Assert.IsTrue(val == 1);
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
            var buffer = new ActionBuffers(ActionSegment<float>.Empty, new ActionSegment<int>(new int[1]));
            m_Adaptor.WriteToHeuristic(m_Action, buffer);
            Assert.IsTrue(buffer.DiscreteActions[0] == 1);
        }
    }
}
#endif // MLA_INPUT_TESTS
