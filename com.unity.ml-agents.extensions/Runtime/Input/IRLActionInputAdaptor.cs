using System;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IRLActionInputAdaptor
    {
        ActionSpec GetActionSpecForInputAction(InputAction action);

        void QueueInputEventForAction(InputAction action, ActionSpec actionSpec, in ActionBuffers actionBuffers);
    }
}
