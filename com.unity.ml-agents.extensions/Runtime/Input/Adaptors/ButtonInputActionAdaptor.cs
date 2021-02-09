#if MLA_INPUT_SYSTEM
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class ButtonInputActionAdaptor : IRLActionInputAdaptor
    {
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeDiscrete(2);
        }

        public void QueueInputEventForAction(InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            var val = actionBuffers.DiscreteActions[0];

            using (StateEvent.From(control.device, out var eventPtr))
            {
                control.WriteValueIntoEvent((float)val, eventPtr);
                InputSystem.QueueEvent(eventPtr);
            }
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var discreteActions = actionBuffers.DiscreteActions;
            var val = action.ReadValue<float>();
            discreteActions[0] = (int)val;
        }
    }
}
#endif // MLA_INPUT_SYSTEM
