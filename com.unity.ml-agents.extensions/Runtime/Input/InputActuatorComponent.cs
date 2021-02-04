using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.Utilities;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActuatorComponent : ActuatorComponent
    {
        InputActionAsset m_InputAsset;
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

        void Start()
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

            m_InputAsset.Disable();
            m_Actuators = GenerateActionActuatorsFromAsset(m_InputAsset, out var layout, IsInHeuristicMode());
            InputSystem.RegisterLayout(layout.ToJson());
            m_Device = CreateDevice(m_InputAsset, m_PlayerInput.defaultActionMap);
            InputSystem.AddDevice(m_Device);
            // Add our device to the device list if there is one.
            if (InputSystem.devices.Count > 0)
            {
                var devices = new List<InputDevice>(InputSystem.devices) { m_Device };
                m_InputAsset.devices = new ReadOnlyArray<InputDevice>(devices.ToArray());
            }
            else
            {
                m_InputAsset.devices = new ReadOnlyArray<InputDevice>(new[] { m_Device });
            }
            // set the device on the actuator so it can get it's control.
            for (var i = 0; i < m_Actuators.Length; i++)
            {
                ((InputActionActuator)m_Actuators[i]).SetDevice(m_Device);
            }
            m_InputAsset.Enable();
            return m_Actuators;
        }

#pragma warning disable 672
        public override IActuator CreateActuator() { return null; }
#pragma warning restore 672

        public override ActionSpec ActionSpec => ActionSpec.MakeContinuous(0);


        static InputDevice CreateDevice(InputActionAsset asset, string defaultMap)
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
            var deviceRequirements = new List<InputControlScheme.DeviceRequirement>();
            deviceRequirements.Add(
                new InputControlScheme.DeviceRequirement
                {
                    controlPath = device.path,
                    isOptional = true,
                    isOR = true
                });

            var map = asset.FindActionMap(defaultMap);
            for (var i = 0; i < map.controlSchemes.Count; i++)
            {
                var cs = map.controlSchemes[i];
                for (var ii = 0; ii < cs.deviceRequirements.Count; ii++)
                {
                    deviceRequirements.Add(cs.deviceRequirements[ii]);
                }
            }

            var inputControlScheme = new InputControlScheme(
                k_MlAgentsControlSchemeName,
                deviceRequirements.ToArray());

            if (asset.FindControlSchemeIndex(k_MlAgentsControlSchemeName) == -1)
            {
                asset.AddControlScheme(inputControlScheme);
            }
            return device;
        }

        IActuator[] GenerateActionActuatorsFromAsset(InputActionAsset asset, out InputControlLayout layout, bool isInHeuristicMode)
        {
            // TODO does this need to change based on the action map we use?
            var builder = new InputControlLayout.Builder()
                .WithName(k_MlAgentsLayoutName)
                .WithFormat(k_MlAgentsLayoutFormat);

            var actuators = new List<IActuator>();

            foreach (var action in asset)
            {
                var actionLayout = InputSystem.LoadLayout(action.expectedControlType);

                builder.AddControl(action.name)
                    .WithLayout(action.expectedControlType)
                    .WithFormat(actionLayout.stateFormat);

                var binding = action.bindings[0];
                var path = $"{k_MlAgentsDevicePath}/{action.name}";
                if (binding.isComposite)
                {
                    var compositeType = s_ControlTypeToCompositeType[action.expectedControlType];
                    action.AddCompositeBinding(compositeType)
                    .With(action.expectedControlType,
                        path,
                        $"{action.bindings[1].groups};{k_MlAgentsControlSchemeName}");
                }
                else
                {
                    action.AddBinding(path,
                        action.interactions,
                        action.processors,
                        $"{binding.groups};{k_MlAgentsControlSchemeName}");
                }

                var adaptor = (IRLActionInputAdaptor)Activator.CreateInstance(
                    s_ControlTypeToAdaptorType[actionLayout.type]);
                actuators.Add(new InputActionActuator(
                    m_Agent,
                    action,
                    adaptor,
                    isInHeuristicMode));

            }
            layout = builder.Build();
            return actuators.ToArray();
        }

        void FindNeededComponents()
        {
            if (m_InputAsset == null)
            {
                var assetProvider = GetComponent<IIntputActionAssetProvider>();
                if (assetProvider != null)
                {
                    m_InputAsset = assetProvider.GetInputActionAsset();
                }
            }
            if (m_PlayerInput == null)
            {
                m_PlayerInput = GetComponent<PlayerInput>();
                Assert.IsNotNull(m_PlayerInput);
                if (m_InputAsset == null)
                {
                    m_InputAsset = m_PlayerInput.actions;
                }
                Assert.IsNotNull(m_InputAsset);
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
            InputSystem.RemoveLayout(k_MlAgentsLayoutName);
            InputSystem.RemoveDevice(m_Device);
            m_InputAsset.RemoveControlScheme(k_MlAgentsControlSchemeName);
            m_InputAsset = null;
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }

    }
}
