using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class Vector2InputActionAdaptor : IRLActionInputAdaptor, IInputHeuristicWriter
    {
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeContinuous(2);
        }

        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers, int continuousOffset, int discreteOffset)
        {
            var value = action.ReadValue<Vector2>();
            var continuousActions = actionBuffers.ContinuousActions;
            continuousActions[continuousOffset++] = value.x;
            continuousActions[continuousOffset] = value.y;
        }

        public void QueueInputEventForAction(InputAction action, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
        }

    }
}
