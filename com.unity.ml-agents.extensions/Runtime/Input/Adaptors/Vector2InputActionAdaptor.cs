#if MLA_INPUT_SYSTEM
using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class Vector2InputActionAdaptor : IRLActionInputAdaptor
    {
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            // TODO create the action spec based on what controls back the action
            return ActionSpec.MakeContinuous(2);
        }

        public void QueueInputEventForAction(InputAction action,
            InputControl control,
            ActionSpec actionSpec,
            in ActionBuffers actionBuffers)
        {
            var x = actionBuffers.ContinuousActions[0];
            var y = actionBuffers.ContinuousActions[1];
            using (StateEvent.From(control.device, out var eventPtr))
            {
                control.WriteValueIntoEvent(new Vector2(x, y), eventPtr);
                InputSystem.QueueEvent(eventPtr);
            }
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var value = action.ReadValue<Vector2>();
            var continuousActions = actionBuffers.ContinuousActions;
            continuousActions[0] = value.x;
            continuousActions[1] = value.y;
        }

    }
}
#endif // MLA_INPUT_SYSTEM
