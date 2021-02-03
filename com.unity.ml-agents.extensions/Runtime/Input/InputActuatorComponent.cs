using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.Utilities;
using UnityEngine.UI;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActuatorComponent : ActuatorComponent
    {
        PlayerInput m_PlayerInput;
        BehaviorParameters m_BehaviorParameters;
        IActuator[] m_Actuators;
        InputDevice m_Device;
        Agent m_Agent;

        static Dictionary<string, string> s_ControlTypeToCompositeType = new Dictionary<string, string>
        {
            {
                "Vector2", "Vector2Value"
            }
        };

        static Dictionary<Type, Type> s_ControlTypeToAdaptorType = new Dictionary<Type, Type>
        {
            { typeof(Vector2Control), typeof(Vector2InputActionAdaptor) },
            { typeof(ButtonControl), typeof(ButtonInputActionAdaptor) }
        };

        const string k_MlAgentsDevicePath = "/MLAgentsLayout";
        const string k_MlAgentsLayoutFormat = "MLAT";
        const string k_MlAgentsLayoutName = "MLAgentsLayout";
        const string k_MlAgentsDeviceName = "MLAgentsDevice";
        const string k_MlAgentsControlSchemeName = "ml-agents";

        void OnEnable()
        {
            FindNeededComponents();
        }

        void OnDisable()
        {
            CleanupActionAsset();
        }

        public override IActuator[] CreateActuators()
        {
            FindNeededComponents();
            var map = m_PlayerInput.actions.FindActionMap(m_PlayerInput.defaultActionMap);
            m_Actuators = GenerateActionActuatorsFromAsset(map, out var layout, IsInHeuristicMode());
            InputSystem.RegisterLayout(layout.ToJson());
            m_Device = CreateDevice(m_PlayerInput, map);
            for (var i = 0; i < m_Actuators.Length; i++)
            {
                ((InputActionActuator)m_Actuators[i]).SetDevice(m_Device);
            }
            return m_Actuators;
        }

#pragma warning disable 672
        public override IActuator CreateActuator() { return null; }
#pragma warning restore 672

        public override ActionSpec ActionSpec => ActionSpec.MakeContinuous(0);


        static InputDevice CreateDevice(PlayerInput playerInput, InputActionMap defaultMap)
        {
            // See if our device layout was already registered.
            var layoutName = InputSystem.TryFindMatchingLayout(new InputDeviceDescription
            {
                interfaceName = k_MlAgentsDeviceName
            });

            // Load the device layout based on the controls we created previously.
            if (string.IsNullOrEmpty(layoutName))
            {
                InputSystem.RegisterLayoutMatcher(k_MlAgentsLayoutName, new InputDeviceMatcher()
                    .WithInterface(k_MlAgentsDeviceName));
            }

            // Actually create the device instance.
            var device = InputSystem.AddDevice(
                new InputDeviceDescription
                {
                    interfaceName = k_MlAgentsDeviceName
                }
            );

            // If the control scheme isn't created, create it with our device registered
            // as required.
            // TODO this may need to be named differently per Agent?
            if (playerInput.actions.FindControlSchemeIndex(k_MlAgentsControlSchemeName) == -1)
            {
                playerInput.actions.AddControlScheme(
                    new InputControlScheme(
                        k_MlAgentsControlSchemeName,
                        new[]
                        {
                            new InputControlScheme.DeviceRequirement
                            {
                                controlPath = device.path,
                                isOptional = false,
                                isOR = true
                            }
                        })
                );
            }

            // Set our device as the only device for this map
            // TODO maybe just add our device to the list?
            defaultMap.devices = new ReadOnlyArray<InputDevice>(new[] { device });
            defaultMap.Enable();
            return device;
        }

        IActuator[] GenerateActionActuatorsFromAsset(InputActionMap actionMap, out InputControlLayout layout, bool isInHeuristicMode)
        {
            // TODO does this need to change based on the action map we use?
            var builder = new InputControlLayout.Builder()
                .WithName(k_MlAgentsLayoutName)
                .WithFormat(k_MlAgentsLayoutFormat);

            var byteOffset = 0;

            var actuators = new IActuator[actionMap.actions.Count];
            var count = 0;

            foreach (var action in actionMap)
            {
                var actionLayout = InputSystem.LoadLayout(action.expectedControlType);

                builder.AddControl(action.name)
                    .WithLayout(action.expectedControlType)
                    .WithByteOffset((uint)byteOffset)
                    .WithFormat(actionLayout.stateFormat);

                var binding = action.bindings[0];
                var path = $"{k_MlAgentsDevicePath}/{action.name}";
                if (binding.isComposite)
                {
                    var compositeType = s_ControlTypeToCompositeType[action.expectedControlType];
                    action.AddCompositeBinding(compositeType)
                    .With(action.expectedControlType,
                        path,
                        k_MlAgentsControlSchemeName);
                }
                else
                {
                    action.AddBinding(path,
                        null,
                        null,
                        k_MlAgentsControlSchemeName);
                }

                byteOffset += actionLayout.stateSizeInBytes;

                var adaptor = (IRLActionInputAdaptor)Activator.CreateInstance(
                    s_ControlTypeToAdaptorType[actionLayout.type]);
                actuators[count++] = new InputActionActuator(
                    m_Agent,
                    action,
                    adaptor,
                    isInHeuristicMode);

            }
            layout = builder.Build();
            return actuators;
        }

        void FindNeededComponents()
        {
            if (m_PlayerInput == null)
            {
                m_PlayerInput = GetComponent<PlayerInput>();
                Assert.IsNotNull(m_PlayerInput);
            }

            if (m_BehaviorParameters == null)
            {
                m_BehaviorParameters = GetComponent<BehaviorParameters>();
                Assert.IsNotNull(m_BehaviorParameters);
            }

            // TODO remove
            if (m_Agent == null)
            {
                m_Agent = GetComponent<Agent>();
            }
        }

        bool IsInHeuristicMode()
        {
            return m_BehaviorParameters.BehaviorType == BehaviorType.HeuristicOnly ||
                     m_BehaviorParameters.BehaviorType == BehaviorType.Default &&
                     ReferenceEquals(m_BehaviorParameters.Model, null);
        }

        void CleanupActionAsset()
        {
            m_PlayerInput.actions.RemoveControlScheme(k_MlAgentsControlSchemeName);
            InputSystem.RemoveLayout(k_MlAgentsLayoutName);
            InputSystem.RemoveDevice(m_Device);
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }

    }
}
