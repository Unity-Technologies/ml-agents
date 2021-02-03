#define ACTUATOR_DEBUG
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

// #undef ACTUATOR_DEBUG

#if ACTUATOR_DEBUG
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.Controls;
using Vector2 = System.Numerics.Vector2;
#endif

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActionActuator : IActuator, IHeuristicProvider
    {
        bool m_IsInHeuristicMode;
        InputAction m_Action;
        IRLActionInputAdaptor m_InputAdaptor;
        InputDevice m_Device;
        InputControl m_Control;
        Agent m_Agent;

#if ACTUATOR_DEBUG
        float m_Time;
        bool m_Flip;
#endif

        public InputActionActuator(Agent agent,
            InputAction action,
            IRLActionInputAdaptor adaptor,
            bool isInHeuristicMode)
        {
            m_Agent = agent;
            m_IsInHeuristicMode = isInHeuristicMode;
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
            if (!m_IsInHeuristicMode)
            {
                m_InputAdaptor.QueueInputEventForAction(m_Action, m_Control, ActionSpec, actionBuffers);
            }
            else
            {
                m_Agent.OnActionReceived(actionBuffers);
            }
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
#if ACTUATOR_DEBUG
            if (Time.realtimeSinceStartup - m_Time > 1.0f)
            {
                m_Flip = !m_Flip;
                m_Time = Time.realtimeSinceStartup;
                if (m_Control.GetType() == typeof(ButtonControl))
                {
                    InputSystem.QueueDeltaStateEvent(m_Control, (byte)1);
                }
            }
            if (m_Control.GetType() == typeof(Vector2Control))
            {
                InputSystem.QueueDeltaStateEvent(m_Control, new Vector2(m_Flip ? 1 : 0, m_Flip ? 0 : 1));
            }
#endif
            m_InputAdaptor.WriteToHeuristic(m_Action, actionBuffersOut);
        }
    }
}
