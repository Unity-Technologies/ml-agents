#if MLA_INPUT_SYSTEM

using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine.InputSystem;
using UnityEngine.Profiling;

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// This implementation of <see cref="IActuator"/> will send events from the ML-Agents training process, or from
    /// neural networks to the <see cref="InputSystem"/> via the <see cref="IRLActionInputAdaptor"/> interface.  If an
    /// <see cref="Agent"/>'s <see cref="BehaviorParameters"/> indicate that the Agent is running in Heuristic Mode,
    /// this Actuator will write actions from the <see cref="InputSystem"/> to the <see cref="ActionBuffers"/> object.
    /// </summary>
    public class InputActionActuator : IActuator, IBuiltInActuator
    {
        readonly BehaviorParameters m_BehaviorParameters;
        readonly InputAction m_Action;
        readonly IRLActionInputAdaptor m_InputAdaptor;
        InputActuatorEventContext m_InputActuatorEventContext;
        InputDevice m_Device;
        InputControl m_Control;

        /// <summary>
        /// Construct an <see cref="InputActionActuator"/> with the <see cref="BehaviorParameters"/> of the
        /// <see cref="Agent"/> component, the relevant <see cref="InputAction"/>, and the relevant
        /// <see cref="IRLActionInputAdaptor"/> to convert between ml-agents &lt;--&gt; <see cref="InputSystem"/>.
        /// </summary>
        /// <param name="inputDevice">The input device this action is bound to.</param>
        /// <param name="behaviorParameters">Used to determine if the <see cref="Agent"/> is running in
        ///     heuristic mode.</param>
        /// <param name="action">The <see cref="InputAction"/> this <see cref="IActuator"/> we read/write data to/from
        ///     via the <see cref="IRLActionInputAdaptor"/>.</param>
        /// <param name="adaptor">The <see cref="IRLActionInputAdaptor"/> that will convert data between ML-Agents
        ///     and the <see cref="InputSystem"/>.</param>
        /// <param name="inputActuatorEventContext">The object that will provide the event ptr to write to.</param>
        public InputActionActuator(InputDevice inputDevice, BehaviorParameters behaviorParameters,
                                   InputAction action,
                                   IRLActionInputAdaptor adaptor,
                                   InputActuatorEventContext inputActuatorEventContext)
        {
            m_BehaviorParameters = behaviorParameters;
            Name = $"InputActionActuator-{action.name}";
            m_Action = action;
            m_InputAdaptor = adaptor;
            m_InputActuatorEventContext = inputActuatorEventContext;
            ActionSpec = adaptor.GetActionSpecForInputAction(m_Action);
            m_Device = inputDevice;
            m_Control = m_Device?.GetChildControl(m_Action.name);
        }

        /// <inheritdoc cref="IActionReceiver.OnActionReceived"/>
        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            Profiler.BeginSample("InputActionActuator.OnActionReceived");
            if (!m_BehaviorParameters.IsInHeuristicMode())
            {
                using (m_InputActuatorEventContext.GetEventForFrame(out var eventPtr))
                {
                    m_InputAdaptor.WriteToInputEventForAction(eventPtr, m_Action, m_Control, ActionSpec, actionBuffers);
                }

            }
            Profiler.EndSample();
        }

        /// <inheritdoc cref="IActionReceiver.WriteDiscreteActionMask"/>
        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            // TODO configure mask from editor UI?
        }

        /// <inheritdoc cref="IActuator.ActionSpec"/>
        public ActionSpec ActionSpec { get; }

        /// <inheritdoc cref="IActuator.Name"/>
        public string Name { get; }

        /// <inheritdoc cref="IActuator.ResetData"/>
        public void ResetData()
        {
            // do nothing for now
        }

        /// <inheritdoc cref="IHeuristicProvider.Heuristic"/>
        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            Profiler.BeginSample("InputActionActuator.Heuristic");
            m_InputAdaptor.WriteToHeuristic(m_Action, actionBuffersOut);
            Profiler.EndSample();
        }

        /// <inheritdoc/>
        public BuiltInActuatorType GetBuiltInActuatorType()
        {
            return BuiltInActuatorType.InputActionActuator;
        }
    }
}

#endif // MLA_INPUT_SYSTEM
