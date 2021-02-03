using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class Vector2InputActionAdaptor : IRLActionInputAdaptor
    {
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            // TODO create the action spec based on what controls back the action
            return ActionSpec.MakeContinuous(2);
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var value = action.ReadValue<Vector2>();
            var continuousActions = actionBuffers.ContinuousActions;
            continuousActions[0] = value.x;
            continuousActions[1] = value.y;
        }

        public void QueueInputEventForAction(InputAction action,
            InputControl control,
            ActionSpec actionSpec,
            in ActionBuffers actionBuffers)
        {
            var x = actionBuffers.ContinuousActions[0];
            var y = actionBuffers.ContinuousActions[1];
            InputSystem.QueueDeltaStateEvent(control, new Vector2(x, y));
        }

    }
}
