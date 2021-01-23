using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

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

        PlayerInput m_playerInput;
        public InputActuator(PlayerInput playerInput)
        {
            Name = "Input System Actuator";
            Debug.Assert(playerInput != null,
                "PlayerInput component is required to use the InputSystemActuator");
            m_playerInput = playerInput;
            ActionSpec = GenerateActionSpecFromAsset(m_playerInput);
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
            return InputSystem.LoadLayout(action.expectedControlType).GetValueType();
        }

        static InputActionMap GetDefaultActionMap(PlayerInput playerInput)
        {
            return playerInput.actions.FindActionMap(playerInput.defaultActionMap);
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            // Queue Input Event (if we aren't in heuristic mode)?
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
            foreach (var action in GetDefaultActionMap(m_playerInput))
            {
                s_HeuristicWriters[GetInputActionValueType(action)].WriteToHeuristic(action, actionBuffersOut, continuousOffset, discreteOffset);
                var spec = s_InputActionToActionSpec[action];
                continuousOffset += spec.NumDiscreteActions;
                discreteOffset += spec.NumDiscreteActions;
            }
        }
    }
}
