using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActuator : IActuator, IHeuristicProvider
    {
        static Dictionary<Type, IRLActionInputAdaptor> s_Adaptors = new Dictionary<Type, IRLActionInputAdaptor>
        {
            {
                typeof(Vector2), new Vector2InputActionAdaptor()
            }
        };

        static Dictionary<Type, IInputHeuristicWriter> s_HeuristicWriters = new Dictionary<Type, IInputHeuristicWriter>
        {
            {
                typeof(Vector2), new Vector2InputActionAdaptor()
            }
        };

        static Dictionary<InputAction, ActionSpec> s_InputActionToActionSpec = new Dictionary<InputAction, ActionSpec>();
        static Dictionary<InputAction, InputControlLayout> s_InputActionToLayout = new Dictionary<InputAction, InputControlLayout>();

        readonly PlayerInput m_PlayerInput;
        readonly BehaviorParameters m_BehaviorParameters;
        readonly Agent m_Agent;
        InputDevice m_VirtualDevice;

        public InputActuator(PlayerInput playerInput, BehaviorParameters behaviorParameters, Agent agent)
        {
            Name = "Input System Actuator";
            Debug.Assert(playerInput != null,
                "PlayerInput component is required to use the InputSystemActuator");
            m_PlayerInput = playerInput;
            m_VirtualDevice = new Gamepad();

            m_BehaviorParameters = behaviorParameters;
            m_Agent = agent;
            ActionSpec = GenerateActionSpecFromAsset(m_PlayerInput);
        }

        static ActionSpec GenerateActionSpecFromAsset(PlayerInput playerInput)
        {
            var actionMap = GetDefaultActionMap(playerInput);
            // var numContinuousActions = 0;

            ActionSpec[] specs = new ActionSpec[actionMap.actions.Count];
            var count = 0;
            foreach (var action in actionMap)
            {
                var valueType = GetInputActionValueType(action);
                var adaptor = s_Adaptors[valueType];
                var spec = adaptor.GetActionSpecForInputAction(action);
                specs[count++] = spec;
                s_InputActionToActionSpec[action] = spec;

            }
            return ActionSpec.Combine(specs);
        }

        static Type GetInputActionValueType(InputAction action)
        {
            if (!s_InputActionToLayout.TryGetValue(action, out var layout))
            {
                layout = InputSystem.LoadLayout(action.expectedControlType);
                s_InputActionToLayout[action] = layout;
            }

            return layout.GetValueType();
        }

        static InputActionMap GetDefaultActionMap(PlayerInput playerInput)
        {
            return playerInput.actions.FindActionMap(playerInput.defaultActionMap);
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            // Queue Input Event (if we aren't in heuristic mode)?
            if (IsInHeuristicMode())
            {
                m_Agent.OnActionReceived(actionBuffers);
            }
            foreach (var action in GetDefaultActionMap(m_PlayerInput))
            {
                // blah
                if (action.activeControl != null)
                {
                    var adaptor = s_Adaptors[GetInputActionValueType(action)];
                    adaptor.QueueInputEventForAction(m_VirtualDevice, action, ActionSpec, actionBuffers);
                }
            }
        }

        bool IsInHeuristicMode()
        {
            return m_BehaviorParameters.BehaviorType == BehaviorType.HeuristicOnly ||
                m_BehaviorParameters.BehaviorType == BehaviorType.Default &&
                ReferenceEquals(m_BehaviorParameters.Model, null);
        }

        bool IsTraining()
        {
            return !IsInHeuristicMode() && Academy.Instance.IsCommunicatorOn;
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            // ???
        }

        public ActionSpec ActionSpec { get; }

        public string Name { get; }

        public void ResetData()
        {

        }

        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            //  Write to actionBuffers
            int continuousOffset = 0;
            int discreteOffset = 0;
            foreach (var action in GetDefaultActionMap(m_PlayerInput))
            {
                s_HeuristicWriters[GetInputActionValueType(action)].WriteToHeuristic(action, actionBuffersOut, continuousOffset, discreteOffset);
                var spec = s_InputActionToActionSpec[action];
                continuousOffset += spec.NumDiscreteActions;
                discreteOffset += spec.NumDiscreteActions;
            }
        }
    }
}
