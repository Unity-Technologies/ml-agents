using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.Utilities;

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

        static Dictionary<string, string> s_ControlTypeToCompositeType = new Dictionary<string, string>
        {
            {
                "Vector2", "Vector2Value"

            }
        };

        static Dictionary<InputAction, ActionSpec> s_InputActionToActionSpec = new Dictionary<InputAction, ActionSpec>();
        static Dictionary<InputAction, InputControlLayout> s_InputActionToLayout = new Dictionary<InputAction, InputControlLayout>();

        readonly PlayerInput m_PlayerInput;
        readonly InputActionMap m_DefaultMap;
        readonly BehaviorParameters m_BehaviorParameters;
        readonly Agent m_Agent;
        InputDevice m_Device;

        const string k_MlAgentsDevicePath = "/MLAgentsLayout*";
        const string k_MlAgentsLayoutFormat = "MLAT";
        const string k_MlAgentsLayoutName = "MLAgentsLayout";
        const string k_MlAgentsDeviceName = "MLAgentsDevice";
        const string k_MlAgentsControlSchemeName = "ml-agents";

        public InputActuator(PlayerInput playerInput, BehaviorParameters behaviorParameters, Agent agent)
        {
            Name = "Input System Actuator";
            Debug.Assert(playerInput != null,
                "PlayerInput component is required to use the InputSystemActuator");
            m_PlayerInput = playerInput;
            var actionAsset = m_PlayerInput.actions;
            m_DefaultMap = actionAsset.FindActionMap(m_PlayerInput.defaultActionMap);

            m_BehaviorParameters = behaviorParameters;
            m_Agent = agent;


            ActionSpec = GenerateActionSpecFromAsset(m_DefaultMap, out var layout, out var groups);
            InputSystem.RegisterLayout(layout.ToJson(), layout.name);

            var layoutName = InputSystem.TryFindMatchingLayout(new InputDeviceDescription
            {
                interfaceName = k_MlAgentsDeviceName
            });

            if (string.IsNullOrEmpty(layoutName))
            {
                InputSystem.RegisterLayoutMatcher(k_MlAgentsLayoutName, new InputDeviceMatcher()
                    .WithInterface(k_MlAgentsDeviceName));
            }

            m_Device = InputSystem.AddDevice(
                new InputDeviceDescription
                {
                    interfaceName = k_MlAgentsDeviceName
                }
            );

            if (playerInput.actions.FindControlSchemeIndex(k_MlAgentsControlSchemeName) == -1)
            {
                playerInput.actions.AddControlScheme(
                    new InputControlScheme(


                        k_MlAgentsControlSchemeName,
                        new[]
                        {
                            new InputControlScheme.DeviceRequirement
                            {
                                controlPath = m_Device.path,
                                isOptional = false,
                                isOR = true
                            }
                        },
                        groups)
                    );
            }
            m_DefaultMap.devices = new ReadOnlyArray<InputDevice>(new[] { m_Device });
            m_DefaultMap.Enable();
        }

        static ActionSpec GenerateActionSpecFromAsset(InputActionMap actionMap, out InputControlLayout layout, out string deviceGroups)
        {
            var specs = new ActionSpec[actionMap.actions.Count];
            var count = 0;
            deviceGroups = default;

            var builder = new InputControlLayout.Builder()
                .WithName(k_MlAgentsLayoutName)
                .WithFormat(k_MlAgentsLayoutFormat);

            var offset = 0;

            foreach (var action in actionMap)
            {
                var actionLayout = GetInputActionLayout(action);
                var compositeType = s_ControlTypeToCompositeType[action.expectedControlType];

                builder.AddControl(action.name)
                    .WithLayout(action.expectedControlType)
                    .WithByteOffset((uint)offset)
                    .WithFormat(actionLayout.stateFormat);

                var binding = action.bindings[0];
                if (binding.isComposite)
                {
                    action.AddCompositeBinding(compositeType)
                    .With(action.expectedControlType,
                        $"{k_MlAgentsDevicePath}/{action.name}",
                        k_MlAgentsControlSchemeName);
                }
                else
                {
                    action.AddBinding($"{action.expectedControlType}:{k_MlAgentsDevicePath}/{action.name}",
                        null,
                        null,
                        k_MlAgentsControlSchemeName);
                }

                offset += actionLayout.stateSizeInBytes;

                var adaptor = s_Adaptors[actionLayout.GetValueType()];
                var spec = adaptor.GetActionSpecForInputAction(action);
                specs[count++] = spec;
                s_InputActionToActionSpec[action] = spec;

            }

            layout = builder.Build();
            return ActionSpec.Combine(specs);
        }

        static InputControlLayout GetInputActionLayout(InputAction action)
        {
            if (!s_InputActionToLayout.TryGetValue(action, out var layout))
            {
                layout = InputSystem.LoadLayout(action.expectedControlType);
                s_InputActionToLayout[action] = layout;
            }

            return layout;
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            // Queue Input Event (if we aren't in heuristic mode)?
            if (IsInHeuristicMode())
            {
                m_Agent.OnActionReceived(actionBuffers);
                return;
            }
            foreach (var action in m_DefaultMap)
            {
                // blah
                if (action.activeControl != null)
                {
                    var adaptor = s_Adaptors[GetInputActionLayout(action).GetValueType()];
                    adaptor.QueueInputEventForAction(action, ActionSpec, actionBuffers);
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

        internal void CleanupActionAsset()
        {
            m_PlayerInput.actions.RemoveControlScheme(k_MlAgentsControlSchemeName);
            InputSystem.RemoveLayout(k_MlAgentsLayoutName);
            InputSystem.RemoveDevice(m_Device);
        }

        public void ResetData()
        {
        }

        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            //  Write to actionBuffers
            var continuousOffset = 0;
            var discreteOffset = 0;
            foreach (var action in m_DefaultMap)
            {
                InputSystem.QueueDeltaStateEvent(m_Device.children[0], new Vector2(0f, 1f));
                var valueType = GetInputActionLayout(action).GetValueType();
                s_HeuristicWriters[valueType].WriteToHeuristic(action, actionBuffersOut, continuousOffset, discreteOffset);
                var spec = s_InputActionToActionSpec[action];
                continuousOffset += spec.NumDiscreteActions;
                discreteOffset += spec.NumDiscreteActions;
            }
        }
    }
}
