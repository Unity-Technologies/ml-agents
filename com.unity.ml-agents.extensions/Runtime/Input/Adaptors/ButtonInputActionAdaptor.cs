#if MLA_INPUT_SYSTEM
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// Class that translates data between the a <see cref="UnityEngine.InputSystem.Controls.ButtonControl"/> and
    /// the ML-Agents <see cref="ActionBuffers"/> object.
    /// </summary>
    public class ButtonInputActionAdaptor : IRLActionInputAdaptor
    {
        /// <summary>
        /// TODO this method needs to be more nuanced depending the types of controls that can back it.  i.e. TriggerControls
        /// are continuous buttons, etc.
        /// Currently returns an <see cref="ActionSpec"/> with 1 branch of size 2.  One value for not pressed, and one
        /// for pressed.
        /// </summary>
        /// <param name="action">The action associated with this adaptor to help determine the action space.</param>
        /// <returns></returns>
        public ActionSpec GetActionSpecForInputAction(InputAction action)
        {
            return ActionSpec.MakeDiscrete(2);
        }

        /// TODO again this might need to be more nuanced for things like continuous buttons.
        /// <inheritdoc cref="IRLActionInputAdaptor.WriteToInputEventForAction"/>
        public void WriteToInputEventForAction(InputEventPtr eventPtr, InputAction action, InputControl control, ActionSpec actionSpec, in ActionBuffers actionBuffers)
        {
            var val = actionBuffers.DiscreteActions[0];
            ((ButtonControl)control).WriteValueIntoEvent((float)val, eventPtr);
        }

        /// <inheritdoc cref="IRLActionInputAdaptor.WriteToHeuristic"/>>
        public void WriteToHeuristic(InputAction action, in ActionBuffers actionBuffers)
        {
            var discreteActions = actionBuffers.DiscreteActions;
            var val = action.ReadValue<float>();
            discreteActions[0] = (int)val;
        }
    }
}
#endif // MLA_INPUT_SYSTEM
