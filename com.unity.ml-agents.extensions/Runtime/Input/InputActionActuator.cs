#define ACTUATOR_DEBUG
#undef ACTUATOR_DEBUG
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;


#if ACTUATOR_DEBUG
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;
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
                    using (StateEvent.From(m_Device, out var eventPtr))
                    {
                        m_Control.WriteValueIntoEvent(m_Flip ? 1f : 0f, eventPtr);
                        InputSystem.QueueEvent(eventPtr);
                        InputSystem.Update();
                    }
                }
            }
            if (m_Control.GetType() == typeof(Vector2Control))
            {
                using (StateEvent.From(m_Device, out var eventPtr))
                {
                    m_Control.WriteValueIntoEvent(new Vector2(m_Flip ? 1 : 0, m_Flip ? 0 : 1), eventPtr);
                    InputSystem.QueueEvent(eventPtr);
                    InputSystem.Update();
                }
            }
#endif
            m_InputAdaptor.WriteToHeuristic(m_Action, actionBuffersOut);
        }
    }
}
