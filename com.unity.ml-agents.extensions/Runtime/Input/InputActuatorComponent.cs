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
    /// <summary>
    /// Component class that handles the parsing of the <see cref="InputActionAsset"/> and translates that into
    /// <see cref="InputActionActuator"/>s.
    /// </summary>
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

        /// <summary>
        /// Mapping of <see cref="InputAction.expectedControlType"/> to a string representing a custom
        /// <see cref="InputBindingComposite{TValue}"/>.
        /// </summary>
        public static Dictionary<string, string> controlTypeToCompositeType = new Dictionary<string, string>
        {
            { "Vector2", "Vector2Value" },
            { "Axis", "AxisValue" }
        };

        /// <summary>
        /// Mapping of <see cref="InputControl"/> types to types of <see cref="IRLActionInputAdaptor"/> concrete classes.
        /// </summary>
        public static Dictionary<Type, Type> controlTypeToAdaptorType = new Dictionary<Type, Type>
        {
            { typeof(Vector2Control), typeof(Vector2InputActionAdaptor) },
            { typeof(ButtonControl), typeof(ButtonInputActionAdaptor) },
            { typeof(IntegerControl), typeof(IntegerInputActionAdaptor) },
            { typeof(AxisControl), typeof(FloatInputActionAdaptor) },
            { typeof(DoubleControl), typeof(DoubleInputActionAdaptor) }
        };

        string m_LayoutName;
        string m_InterfaceName;

        public const string mlAgentsLayoutFormat = "MLAT";
        public const string mlAgentsLayoutName = "MLAgentsLayout";
        public const string mlAgentsDeviceName = "MLAgentsDevice";
        public const string mlAgentsControlSchemeName = "ml-agents";

        void Start()
        {
            FindNeededComponents();
        }

        void OnDisable()
        {
            CleanupActionAsset();
        }

        /// <summary>
        /// This method is where the <see cref="InputActionAsset"/> gets parsed and translated into
        /// <see cref="InputActionActuator"/>s that communicate with the <see cref="InputSystem"/> via a
        /// virtual <see cref="InputDevice"/>.
        /// <remarks>
        /// The flow of this method is as follows:
        /// <list type="number">
        ///     <item>
        ///     <description>Ensure that our custom <see cref="InputBindingComposite{TValue}"/>s are registered with
        ///     the InputSystem.</description>
        ///     </item>
        ///     <item>
        ///     <description>Look for the components that are needed by this class in order to retrieve the
        ///     <see cref="InputActionAsset"/>.  It first looks for <see cref="IInputActionAssetProvider"/>, if that
        ///     is not found, it will get the asset from the <see cref="PlayerInput"/> component.</description>
        ///     </item>
        ///     <item>
        ///     <description>Create the list <see cref="InputActionActuator"/>s, one for each action in the default
        ///     <see cref="InputActionMap"/> as set by the <see cref="PlayerInput"/> component.  Within the method
        ///     where the actuators are being created, an <see cref="InputControlLayout"/> is also being built based
        ///     on the number and types of <see cref="InputAction"/>s.  This will be used to create a virtual
        ///     <see cref="InputDevice"/> with a <see cref="InputControlLayout"/> that is specific to the
        ///     <see cref="InputActionMap"/> specified by <see cref="PlayerInput"/></description>
        ///     </item>
        ///     <item>
        ///     <description>Create our device based on the layout that was generated and registered during
        ///     actuator creation.</description>
        ///     </item>
        ///     <item>
        ///     <description>Create an ml-agents control scheme and add it to the <see cref="InputActionAsset"/> so
        ///     our virtual devices can be used.</description>
        ///     </item>
        ///     <item>
        ///     <description>Add our virtual <see cref="InputDevice"/> to the input system.</description>
        ///     </item>
        /// </list>
        /// </remarks>
        /// </summary>
        /// <returns>A list of </returns>
        public override IActuator[] CreateActuators()
        {
            // TODO cgoy better way to create ids?
            m_LocalId = s_DeviceId++;

            // RuntimeInitializeOnLoadMethod was not sufficient to have our composites loaded
            // by the time this method was called.  So this is a workaround to ensure that
            // our custom composites get registered with the InputSystem.
            InputCompositeLoader.Init();

            FindNeededComponents();

            m_InputAsset.Disable();

            m_Actuators = GenerateActionActuatorsFromAsset(
                m_InputAsset,
                m_LayoutName,
                m_LocalId,
                m_BehaviorParameters);

            m_Device = CreateDevice(m_LayoutName, m_InterfaceName, m_LocalId);

            CreateControlScheme(m_InputAsset, m_PlayerInput.defaultActionMap, m_Device);

            InputSystem.AddDevice(m_Device);

            for (var i = 0; i < m_Actuators.Length; i++)
            {
                ((InputActionActuator)m_Actuators[i]).SetDevice(m_Device);
            }

            m_InputAsset.Enable();
            return m_Actuators;
        }

        /// <summary>
        /// This method creates a control scheme we can add our device to in order for it to be discovered by
        /// the <see cref="InputSystem"/>.
        /// </summary>
        /// <param name="asset">The <see cref="InputAction"/> to add the <see cref="InputControlScheme"/> to.</param>
        /// <param name="defaultMap">The <see cref="InputActionMap"/> to pull
        /// <see cref="InputControlScheme.DeviceRequirement"/>s from</param>
        /// <param name="device">The virtual device to add to our custom control scheme.</param>
        internal static void CreateControlScheme(InputActionAsset asset, string defaultMap, InputControl device)
        {
            // If the control scheme isn't created, create it with our device registered
            // as required.  Devices with the same path get incremented automatically by the InputSystem
            // after the first one.
            var deviceRequirements = new List<InputControlScheme.DeviceRequirement>
            {
                new InputControlScheme.DeviceRequirement
                {
                    controlPath = device.path,
                    isOptional = true,
                    isOR = true
                }
            };

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
                mlAgentsControlSchemeName,
                deviceRequirements.ToArray());

            if (asset.FindControlSchemeIndex(mlAgentsControlSchemeName) == -1)
            {
                asset.AddControlScheme(inputControlScheme);
            }
        }

#pragma warning disable 672
        /// <inheritdoc cref="ActuatorComponent.CreateActuator"/>
        public override IActuator CreateActuator() { return null; }
#pragma warning restore 672

        /// <inheritdoc cref="IActuator.ActionSpec"/>
        public override ActionSpec ActionSpec => ActionSpec.MakeContinuous(0);


        /// <summary>
        /// Creates a virtual <see cref="InputDevice"/> with a layout that is based on the <see cref="InputAction"/>s
        /// in the <see cref="InputActionMap"/> specified by <see cref="PlayerInput.defaultActionMap"/>.
        /// </summary>
        /// <param name="layoutName">The custom layout name to use for the <see cref="InputDevice"/> to create.</param>
        /// <param name="interfaceName">The made up interface name we are using for these virtual devices.  Right now
        /// it is equal to <see cref="mlAgentsDeviceName"/> + <see cref="BehaviorParameters.BehaviorName"/></param>
        /// <param name="localId"></param>
        /// <returns></returns>
        internal static InputDevice CreateDevice(string layoutName, string interfaceName, uint localId)
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
            return device;
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="asset"></param>
        /// <param name="layoutName"></param>
        /// <param name="localId"></param>
        /// <param name="behaviorParameters"></param>
        /// <returns></returns>
        internal static IActuator[] GenerateActionActuatorsFromAsset(
            InputActionAsset asset,
            string layoutName,
            uint localId,
            BehaviorParameters behaviorParameters)
        {
            // TODO does this need to change based on the action map we use?
            var builder = new InputControlLayout.Builder()
                .WithName(layoutName)
                .WithFormat(mlAgentsLayoutFormat);

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
                    var compositeType = controlTypeToCompositeType[action.expectedControlType];
                    action.AddCompositeBinding(compositeType)
                    .With(action.expectedControlType,
                        path,
                        $"{child.groups}{InputBinding.Separator}{mlAgentsControlSchemeName}");
                }
                else
                {
                    action.AddBinding(path,
                        action.interactions,
                        action.processors,
                        $"{binding.groups}{InputBinding.Separator}{mlAgentsControlSchemeName}");
                }

                var adaptor = (IRLActionInputAdaptor)Activator.CreateInstance(
                    controlTypeToAdaptorType[actionLayout.type]);
                actuators.Add(new InputActionActuator(
                    behaviorParameters,
                    action,
                    adaptor));

            }
            var layout = builder.Build();
            if (InputSystem.LoadLayout(layout.name) == null)
            {
                InputSystem.RegisterLayout(layout.ToJson(), layout.name);
            }
            return actuators.ToArray();
        }

        internal void FindNeededComponents()
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
                m_LayoutName = mlAgentsLayoutName + m_BehaviorParameters.BehaviorName;
                m_InterfaceName = mlAgentsDeviceName + m_BehaviorParameters.BehaviorName;
            }

            // TODO remove
            if (m_Agent == null)
            {
                m_Agent = GetComponent<Agent>();
            }
        }

        internal void CleanupActionAsset()
        {
            InputSystem.RemoveLayout(mlAgentsLayoutName);
            if (m_Device != null)
            {
                InputSystem.RemoveDevice(m_Device);
            }
            m_InputAsset.RemoveControlScheme(mlAgentsControlSchemeName);
            m_InputAsset = null;
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }

    }
}
#endif // MLA_INPUT_SYSTEM
