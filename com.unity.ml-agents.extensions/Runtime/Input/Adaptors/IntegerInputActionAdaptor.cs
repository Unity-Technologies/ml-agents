#if MLA_INPUT_SYSTEM
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class IntegerInputActionAdaptor : IRLActionInputAdaptor
    {
        // TODO need to figure out how we can infer the branch size from here.
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeDiscrete(2);
        }

        public void QueueInputEventForAction(InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            var val = actionBuffers.DiscreteActions[0];

            using (StateEvent.From(control.device, out var eventPtr))
            {
                control.WriteValueIntoEvent(val, eventPtr);
                InputSystem.QueueEvent(eventPtr);
            }
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var actions = actionBuffers.DiscreteActions;
            var val = action.ReadValue<int>();
            actions[0] = val;
        }
    }
}
#endif // MLA_INPUT_SYSTEM
