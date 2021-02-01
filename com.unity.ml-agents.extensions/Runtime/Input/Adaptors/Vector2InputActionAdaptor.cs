using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Composites;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.LowLevel;

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
            // Debug.Assert(action.controls.Count == 1, "Vector2InputActionAdaptor should have exactly one control.");
            var value = action.ReadValue<Vector2>();
            var continuousActions = actionBuffers.ContinuousActions;
            continuousActions[continuousOffset++] = value.x;
            continuousActions[continuousOffset] = value.y;
        }

        public void QueueInputEventForAction(InputAction action, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            // Debug.Assert(action.controls.Count == 1, "Vector2InputActionAdaptor should have exactly one control.");
            var x = actionBuffers.ContinuousActions[0];
            var y = actionBuffers.ContinuousActions[1];
            var control = action.activeControl;
            InputSystem.QueueDeltaStateEvent(control, new Vector2(x, y));
        }

    }
}
