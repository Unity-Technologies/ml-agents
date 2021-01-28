using System;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IRLActionInputAdaptor
    {
        ActionSpec GetActionSpecForInputAction(InputAction action);

        void QueueInputEventForAction(InputDevice device, InputAction action, ActionSpec actionSpec, in ActionBuffers actionBuffers);
    }
}
