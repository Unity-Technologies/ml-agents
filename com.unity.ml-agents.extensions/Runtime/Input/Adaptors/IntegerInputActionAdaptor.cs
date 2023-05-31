#if MLA_INPUT_SYSTEM
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// Translates data from a <see cref="UnityEngine.InputSystem.Controls.IntegerControl"/>.
    /// </summary>
    public class IntegerInputActionAdaptor : IRLActionInputAdaptor
    {
        // TODO need to figure out how we can infer the branch size from here.
        /// <inheritdoc cref="IRLActionInputAdaptor.GetActionSpecForInputAction"/>
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeDiscrete(2);
        }

        /// <inheritdoc cref="IRLActionInputAdaptor.WriteToInputEventForAction"/>
        public void WriteToInputEventForAction(InputEventPtr eventPtr, InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            var val = actionBuffers.DiscreteActions[0];
            control.WriteValueIntoEvent(val, eventPtr);
        }

        /// <inheritdoc cref="IRLActionInputAdaptor.WriteToHeuristic"/>
        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var actions = actionBuffers.DiscreteActions;
            var val = action.ReadValue<int>();
            actions[0] = val;
        }
    }
}
#endif // MLA_INPUT_SYSTEM
