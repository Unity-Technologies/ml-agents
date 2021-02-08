#if MLA_INPUT_SYSTEM
using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Runtime.Input.Composites;
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
        uint m_LocalId;

        static uint s_DeviceId;

        static Dictionary<string, string> s_ControlTypeToCompositeType = new Dictionary<string, string>
        {
            { "Vector2", "Vector2Value" },
            { "Axis", "AxisValue" }
        };

        static Dictionary<Type, Type> s_ControlTypeToAdaptorType = new Dictionary<Type, Type>
        {
            { typeof(Vector2Control), typeof(Vector2InputActionAdaptor) },
            { typeof(ButtonControl), typeof(ButtonInputActionAdaptor) },
            { typeof(int), typeof(IntegerInputActionAdaptor) },
            { typeof(float), typeof(FloatInputActionAdaptor) },
            { typeof(double), typeof(DoubleInputActionAdaptor) }
        };

        string m_LayoutName;
        string m_InterfaceName;

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
            InputCompositeLoader.Init();
            FindNeededComponents();

            m_LocalId = s_DeviceId++;

            m_InputAsset.Disable();
            m_Actuators = GenerateActionActuatorsFromAsset(
                m_InputAsset,
                m_LayoutName,
                m_LocalId,
                m_BehaviorParameters,
                out var layout);

            if (InputSystem.LoadLayout(layout.name) == null)
            {
                InputSystem.RegisterLayout(layout.ToJson(), layout.name);
            }

            m_Device = CreateDevice(m_InputAsset,
                m_LayoutName,
                m_InterfaceName,
                m_PlayerInput.defaultActionMap,
                m_LocalId);

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


        static InputDevice CreateDevice(InputActionAsset asset, string layoutName, string interfaceName, string defaultMap, uint localId)
        {
            // See if our device layout was already registered.
            var existingLayout = InputSystem.TryFindMatchingLayout(new InputDeviceDescription
            {
                interfaceName = interfaceName
            });

            // Load the device layout based on the controls we created previously.
            if (string.IsNullOrEmpty(existingLayout))
            {
                InputSystem.RegisterLayoutMatcher(layoutName, new InputDeviceMatcher()
                    .WithInterface($"^({interfaceName}[0-9]*)"));
            }

            // Actually create the device instance.
            var device = InputSystem.AddDevice(
                new InputDeviceDescription
                {
                    interfaceName = interfaceName,
                    serial = $"{localId}"
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

        static IActuator[] GenerateActionActuatorsFromAsset(
            InputActionAsset asset,
            string layoutName,
            uint localId,
            BehaviorParameters behaviorParameters,
            out InputControlLayout layout)
        {
            // TODO does this need to change based on the action map we use?
            var builder = new InputControlLayout.Builder()
                .WithName(layoutName)
                .WithFormat(k_MlAgentsLayoutFormat);

            var actuators = new List<IActuator>();

            foreach (var action in asset)
            {

                var actionLayout = InputSystem.LoadLayout(action.expectedControlType);

                builder.AddControl(action.name)
                    .WithLayout(action.expectedControlType)
                    .WithFormat(actionLayout.stateFormat);

                var binding = action.bindings[0];
                var devicePath = InputControlPath.Separator + layoutName;

                // Reasonably, the input system starts adding numbers after the first none numbered name
                // is added.  So for device ID of 0, we use the empty string in the path.
                var deviceId = localId == 0 ? string.Empty : "" + localId;
                var path = $"{devicePath}{deviceId}{InputControlPath.Separator}{action.name}";
                if (binding.isComposite)
                {
                    // search for a child of the composite so we can get the groups
                    // this is not technically correct as each binding can have different groups
                    InputBinding child = action.bindings[1];
                    for (var i = 1; i < action.bindings.Count; i++)
                    {
                        var candidate = action.bindings[i];
                        if (candidate.isComposite || binding.action != candidate.action)
                        {
                            continue;
                        }

                        child = candidate;
                        break;

                    }
                    var compositeType = s_ControlTypeToCompositeType[action.expectedControlType];
                    action.AddCompositeBinding(compositeType)
                    .With(action.expectedControlType,
                        path,
                        $"{child.groups}{InputBinding.Separator}{k_MlAgentsControlSchemeName}");
                }
                else
                {
                    action.AddBinding(path,
                        action.interactions,
                        action.processors,
                        $"{binding.groups}{InputBinding.Separator}{k_MlAgentsControlSchemeName}");
                }

                var adaptor = (IRLActionInputAdaptor)Activator.CreateInstance(
                    s_ControlTypeToAdaptorType[actionLayout.type]);
                actuators.Add(new InputActionActuator(
                    behaviorParameters,
                    action,
                    adaptor));

            }
            layout = builder.Build();
            return actuators.ToArray();
        }

        void FindNeededComponents()
        {
            if (m_InputAsset == null)
            {
                var assetProvider = GetComponent<IInputActionAssetProvider>();
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
                m_LayoutName = k_MlAgentsLayoutName + m_BehaviorParameters.BehaviorName;
                m_InterfaceName = k_MlAgentsDeviceName + m_BehaviorParameters.BehaviorName;
            }

            // TODO remove
            if (m_Agent == null)
            {
                m_Agent = GetComponent<Agent>();
            }
        }

        void CleanupActionAsset()
        {
            InputSystem.RemoveLayout(k_MlAgentsLayoutName);
            if (m_Device != null)
            {
                InputSystem.RemoveDevice(m_Device);
            }
            m_InputAsset.RemoveControlScheme(k_MlAgentsControlSchemeName);
            m_InputAsset = null;
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }

    }
}
#endif // MLA_INPUT_SYSTEM
