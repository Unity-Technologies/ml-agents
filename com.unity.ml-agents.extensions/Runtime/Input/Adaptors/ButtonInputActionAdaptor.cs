using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class ButtonInputActionAdaptor : IRLActionInputAdaptor
    {
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeDiscrete(1);
        }

        public void QueueInputEventForAction(InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            var val = actionBuffers.DiscreteActions[0];
            InputSystem.QueueDeltaStateEvent(control, (byte)val);
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var discreteActions = actionBuffers.DiscreteActions;
            var val = action.ReadValue<byte>();
            discreteActions[0] = val;
        }
    }
}
