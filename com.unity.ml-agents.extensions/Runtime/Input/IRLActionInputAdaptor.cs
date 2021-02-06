#if MLA_INPUT_SYSTEM
using System;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IRLActionInputAdaptor
    {
        ActionSpec GetActionSpecForInputAction(InputAction action);

        void QueueInputEventForAction(InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers);

        void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers);
    }
}
#endif // MLA_INPUT_SYSTEM
