#if MLA_INPUT_SYSTEM

using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActionActuator : IActuator, IHeuristicProvider
    {
        readonly BehaviorParameters m_BehaviorParameters;
        InputAction m_Action;
        IRLActionInputAdaptor m_InputAdaptor;
        InputDevice m_Device;
        InputControl m_Control;

        public InputActionActuator(
            BehaviorParameters behaviorParameters,
            InputAction action,
            IRLActionInputAdaptor adaptor)
        {
            m_BehaviorParameters = behaviorParameters;
            Name = $"InputActionActuator-{action.name}";
            m_Action = action;
            m_InputAdaptor = adaptor;
            ActionSpec = adaptor.GetActionSpecForInputAction(m_Action);
        }

        internal void SetDevice(InputDevice device)
        {
            m_Device = device;
            m_Control = m_Device.GetChildControl(m_Action.name);
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (!IsInHeuristicMode())
            {
                m_InputAdaptor.QueueInputEventForAction(m_Action, m_Control, ActionSpec, actionBuffers);
            }
        }

        bool IsInHeuristicMode()
        {
            return m_BehaviorParameters.BehaviorType == BehaviorType.HeuristicOnly ||
                     m_BehaviorParameters.BehaviorType == BehaviorType.Default &&
                     ReferenceEquals(m_BehaviorParameters.Model, null) &&
                     !Academy.Instance.IsCommunicatorOn;
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            // TODO configure mask from editor UI?
        }

        public ActionSpec ActionSpec { get; }
        public string Name { get; }

        public void ResetData()
        {
            // do nothing for now
        }

        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            m_InputAdaptor.WriteToHeuristic(m_Action, actionBuffersOut);
        }
    }
}

#endif // MLA_INPUT_SYSTEM
