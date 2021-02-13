#if MLA_INPUT_SYSTEM && UNITY_2019_4_OR_NEWER
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
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// Component class that handles the parsing of the <see cref="InputActionAsset"/> and translates that into
    /// <see cref="InputActionActuator"/>s.
    /// </summary>
    public class InputActuatorComponent : ActuatorComponent
    {
        InputActionAsset m_InputAsset;
        IInputActionCollection2 m_InputAssetCollection;
        PlayerInput m_PlayerInput;
        BehaviorParameters m_BehaviorParameters;
        IActuator[] m_Actuators;
        InputDevice m_Device;
        Agent m_Agent;
        uint m_LocalId;

        static uint s_DeviceId;

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
        ActionSpec m_ActionSpec;

        public const string mlAgentsLayoutFormat = "MLAT";
        public const string mlAgentsLayoutName = "MLAgentsLayout";
        public const string mlAgentsDeviceName = "MLAgentsDevice";
        public const string mlAgentsControlSchemeName = "ml-agents";

        void Awake()
        {
            // TODO cgoy better way to create ids?
            m_LocalId = s_DeviceId++;
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
            FindNeededComponents();
            m_InputAsset.Disable();
            var inputActionMap = m_InputAsset.FindActionMap(m_PlayerInput.defaultActionMap);

            m_Actuators = GenerateActionActuatorsFromAsset(
                inputActionMap,
                m_LayoutName,
                m_LocalId,
                m_BehaviorParameters);

            m_Device = CreateDevice(m_LayoutName, m_InterfaceName, m_LocalId);
            var controlScheme = CreateControlScheme(m_InputAsset, m_Device, m_LocalId);
            if (m_InputAsset.FindControlSchemeIndex(controlScheme.name) == -1)
            {
                m_InputAsset.AddControlScheme(controlScheme);
            }

            m_InputAssetCollection.bindingMask = InputBinding.MaskByGroup(controlScheme.bindingGroup);
            InputSystem.AddDevice(m_Device);
            m_InputAssetCollection.devices = new ReadOnlyArray<InputDevice>(new[] { m_Device });
            inputActionMap.devices = m_InputAssetCollection.devices;
            inputActionMap.Enable();

            var specs = new ActionSpec[m_Actuators.Length];
            for (var i = 0; i < m_Actuators.Length; i++)
            {
                var actuator = m_Actuators[i];
                ((InputActionActuator)actuator).SetDevice(m_Device);
                specs[i] = actuator.ActionSpec;
            }


            m_ActionSpec = ActionSpec.Combine(specs);
            m_InputAsset.Enable();
            return m_Actuators;
        }

        /// <summary>
        /// This method creates a control scheme and adds it to the <see cref="InputActionAsset"/> passed in so
        /// we can add our device to in order for it to be discovered by the <see cref="InputSystem"/>.
        /// </summary>
        /// <param name="asset">The <see cref="InputAction"/> to add the <see cref="InputControlScheme"/> to.</param>
        /// <param name="device">The virtual device to add to our custom control scheme.</param>
        /// <param name="localId">id of device</param>
        internal static InputControlScheme CreateControlScheme(InputActionAsset asset, InputControl device, uint localId)
        {
            // If the control scheme isn't created, create it with our device registered
            // as required.  Devices with the same path get incremented automatically by the InputSystem
            // after the first one.
            var deviceRequirements = new[]
            {
                new InputControlScheme.DeviceRequirement
                {
                    controlPath = device.path,
                    isOptional = true,
                    isAND = false,
                    isOR = true
                }
            };
            var controlSchemeName = mlAgentsControlSchemeName + localId;
            var inputControlScheme = new InputControlScheme(
                controlSchemeName,
                deviceRequirements);

            return inputControlScheme;
        }

#pragma warning disable 672
        /// <inheritdoc cref="ActuatorComponent.CreateActuator"/>
        public override IActuator CreateActuator() { return null; }
#pragma warning restore 672

        /// <inheritdoc cref="IActuator.ActionSpec"/>
        public override ActionSpec ActionSpec => m_ActionSpec;

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
        /// <param name="defaultMap"></param>
        /// <param name="layoutName"></param>
        /// <param name="localId"></param>
        /// <param name="behaviorParameters"></param>
        /// <returns></returns>
        internal static IActuator[] GenerateActionActuatorsFromAsset(
            InputActionMap defaultMap,
            string layoutName,
            uint localId,
            BehaviorParameters behaviorParameters)
        {
            // TODO does this need to change based on the action map we use?
            var builder = new InputControlLayout.Builder()
                .WithName(layoutName)
                .WithFormat(mlAgentsLayoutFormat);

            var actuators = new List<IActuator>();

            foreach (var action in defaultMap)
            {
                var actionLayout = InputSystem.LoadLayout(action.expectedControlType);

                builder.AddControl(action.name)
                    .WithLayout(action.expectedControlType)
                    .WithFormat(actionLayout.stateFormat)
                    .WithUsages($"MLAgentPlayer-{localId}");

                var binding = action.bindings[0];
                var devicePath = InputControlPath.Separator + layoutName;

                // Reasonably, the input system starts adding numbers after the first none numbered name
                // is added.  So for device ID of 0, we use the empty string in the path.
                var deviceId = localId == 0 ? string.Empty : "" + localId;
                var path = $"{devicePath}{deviceId}{InputControlPath.Separator}{action.name}";
                // if (binding.isComposite)
                // {
                //     // search for a child of the composite so we can get the groups
                //     // this is not technically correct as each binding can have different groups
                //     InputBinding? child = null;
                //     for (var i = 1; i < action.bindings.Count; i++)
                //     {
                //         var candidate = action.bindings[i];
                //         if (candidate.isComposite || binding.action != candidate.action)
                //         {
                //             continue;
                //         }
                //
                //         child = candidate;
                //         break;
                //     }
                //
                //     var groups = string.Empty;
                //     // if (child != null)
                //     // {
                //     //     groups = child.Value.groups;
                //     // }
                //     // var compositeType = controlTypeToCompositeType[action.expectedControlType];
                //     action.AddBinding(path,
                //         action.interactions,
                //         action.processors,
                //          mlAgentsControlSchemeName);
                //         // $"{groups}{InputBinding.Separator}{mlAgentsControlSchemeName}");
                // }
                // else
                // {
                action.AddBinding(path,
                    action.interactions,
                    action.processors,
                    mlAgentsControlSchemeName + localId);
                // }

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
                Assert.IsNotNull(assetProvider);
                (m_InputAsset, m_InputAssetCollection) = assetProvider.GetInputActionAsset();
                Assert.IsNotNull(m_InputAsset, "An InputActionAsset could not be found on IInputActionAssetProvider or PlayerInput.");
                Assert.IsNotNull(m_InputAssetCollection, "An IInputActionCollection2 could not be found on IInputActionAssetProvider or PlayerInput.");
            }
            if (m_PlayerInput == null)
            {
                m_PlayerInput = GetComponent<PlayerInput>();
                Assert.IsNotNull(m_PlayerInput, "PlayerInput component could not be found on this GameObject.");
            }

            if (m_BehaviorParameters == null)
            {
                m_BehaviorParameters = GetComponent<BehaviorParameters>();
                Assert.IsNotNull(m_BehaviorParameters, "BehaviorParameters were not on the current GameObject.");
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

            if (m_InputAsset != null
                && m_InputAsset.FindControlSchemeIndex(mlAgentsControlSchemeName + m_LocalId) != -1)
            {
                m_InputAsset.RemoveControlScheme(mlAgentsControlSchemeName + m_LocalId);
            }

            if (m_Actuators != null)
            {
                Array.Clear(m_Actuators, 0, m_Actuators.Length);
            }
            m_InputAsset = null;
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }
    }
}
#endif // MLA_INPUT_SYSTEM && UNITY_2019_4_OR_NEWER
