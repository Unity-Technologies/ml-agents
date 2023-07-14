#if MLA_INPUT_SYSTEM
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;
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
    [RequireComponent(typeof(PlayerInput), typeof(IInputActionAssetProvider))]
    [AddComponentMenu("ML Agents/Input Actuator", (int)MenuGroup.Actuators)]
    public class InputActuatorComponent : ActuatorComponent
    {
        InputActionAsset m_InputAsset;
        IInputActionCollection2 m_AssetCollection;
        PlayerInput m_PlayerInput;
        BehaviorParameters m_BehaviorParameters;
        IActuator[] m_Actuators;
        InputDevice m_Device;

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
        [SerializeField]
        ActionSpec m_ActionSpec;
        InputControlScheme m_ControlScheme;

        public const string mlAgentsLayoutFormat = "MLAT";
        public const string mlAgentsLayoutName = "MLAgentsLayout";
        public const string mlAgentsControlSchemeName = "ml-agents";

        /// <inheritdoc cref="IActuator.ActionSpec"/>
        public override ActionSpec ActionSpec
        {
            get
            {
#if UNITY_EDITOR
                if (!EditorApplication.isPlaying && m_ActionSpec.NumContinuousActions == 0
                    && m_ActionSpec.BranchSizes == null
                    || m_ActionSpec.BranchSizes.Length == 0)
                {
                    FindNeededComponents();
                    var actuators = CreateActuatorsFromMap(m_InputAsset.FindActionMap(m_PlayerInput.defaultActionMap),
                        m_BehaviorParameters,
                        null,
                        InputActuatorEventContext.s_EditorContext);
                    m_ActionSpec = CombineActuatorActionSpecs(actuators);
                }
#endif
                return m_ActionSpec;
            }
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
            var collection = m_AssetCollection ?? m_InputAsset;
            collection.Disable();
            var inputActionMap = m_InputAsset.FindActionMap(m_PlayerInput.defaultActionMap);

            RegisterLayoutBuilder(inputActionMap, m_LayoutName);
            m_Device = InputSystem.AddDevice(m_LayoutName);

            var context = new InputActuatorEventContext(inputActionMap.actions.Count, m_Device);
            m_Actuators = CreateActuatorsFromMap(inputActionMap, m_BehaviorParameters, m_Device, context);

            UpdateDeviceBinding(m_BehaviorParameters.IsInHeuristicMode());
            inputActionMap.Enable();

            m_ActionSpec = CombineActuatorActionSpecs(m_Actuators);
            collection.Enable();
            return m_Actuators;
        }

        static ActionSpec CombineActuatorActionSpecs(IActuator[] actuators)
        {
            var specs = new ActionSpec[actuators.Length];
            for (var i = 0; i < actuators.Length; i++)
            {
                specs[i] = actuators[i].ActionSpec;
            }
            return ActionSpec.Combine(specs);
        }

        internal static IActuator[] CreateActuatorsFromMap(InputActionMap inputActionMap,
            BehaviorParameters behaviorParameters,
            InputDevice inputDevice,
            InputActuatorEventContext context)
        {
            var actuators = new IActuator[inputActionMap.actions.Count];
            for (var i = 0; i < inputActionMap.actions.Count; i++)
            {
                var action = inputActionMap.actions[i];
                var actionLayout = InputSystem.LoadLayout(action.expectedControlType);
                var adaptor = (IRLActionInputAdaptor)Activator.CreateInstance(controlTypeToAdaptorType[actionLayout.type]);
                actuators[i] = new InputActionActuator(inputDevice, behaviorParameters, action, adaptor, context);

                // Reasonably, the input system starts adding numbers after the first none numbered name
                // is added.  So for device ID of 0, we use the empty string in the path.
                var path = $"{inputDevice?.path}{InputControlPath.Separator}{action.name}";
                action.AddBinding(path,
                    action.interactions,
                    action.processors,
                    mlAgentsControlSchemeName);
                action.bindingMask = InputBinding.MaskByGroup(mlAgentsControlSchemeName);
            }
            return actuators;
        }

        /// <summary>
        /// Set up bindings based on whether or not the BehaviorParameters are working in Heuristic mode or not.
        /// If we are working in Heuristic mode, we want the input system to handle everything.  If not, we
        /// want the neural network to send input from virtual devices.
        /// </summary>
        /// <param name="isInHeuristicMode">true if the Agent connected to this GameObject is working in
        /// Heuristic mode.</param>
        /// <seealso cref="BehaviorParameters.IsInHeuristicMode"/>
        internal void UpdateDeviceBinding(bool isInHeuristicMode)
        {
            if (ReferenceEquals(m_Device, null))
            {
                return;
            }
            var collection = m_AssetCollection ?? m_InputAsset;
            m_ControlScheme = CreateControlScheme(m_Device, isInHeuristicMode, m_InputAsset);
            if (m_InputAsset.FindControlSchemeIndex(m_ControlScheme.name) != -1)
            {
                m_InputAsset.RemoveControlScheme(m_ControlScheme.name);
            }

            if (!isInHeuristicMode)
            {
                var inputActionMap = m_InputAsset.FindActionMap(m_PlayerInput.defaultActionMap);
                m_InputAsset.AddControlScheme(m_ControlScheme);
                collection.bindingMask = InputBinding.MaskByGroup(m_ControlScheme.bindingGroup);
                collection.devices = new ReadOnlyArray<InputDevice>(new[] { m_Device });
                inputActionMap.bindingMask = collection.bindingMask;
                inputActionMap.devices = collection.devices;
            }
            else
            {
                var inputActionMap = m_InputAsset.FindActionMap(m_PlayerInput.defaultActionMap);
                collection.bindingMask = null;
                collection.devices = InputSystem.devices;
                inputActionMap.devices = InputSystem.devices;
                inputActionMap.bindingMask = null;
            }
            collection.Enable();
        }

        /// <summary>
        /// This method creates a control scheme and adds it to the <see cref="InputActionAsset"/> passed in so
        /// we can add our device to in order for it to be discovered by the <see cref="InputSystem"/>.
        /// </summary>
        /// <param name="device">The virtual device to add to our custom control scheme.</param>
        /// <param name="isInHeuristicMode">if we are in heuristic mode, we need to add other other device requirements.</param>
        /// <param name="asset">The InputActionAsset to get the device requirements from</param>
        internal static InputControlScheme CreateControlScheme(InputControl device,
            bool isInHeuristicMode,
            InputActionAsset asset)
        {
            var deviceRequirements = new List<InputControlScheme.DeviceRequirement>
            {
                new InputControlScheme.DeviceRequirement
                {
                    controlPath = InputBinding.Separator + mlAgentsLayoutName
                }
            };

            if (isInHeuristicMode)
            {
                for (var i = 0; i < asset.controlSchemes.Count; i++)
                {
                    var scheme = asset.controlSchemes[i];
                    for (var ii = 0; ii < scheme.deviceRequirements.Count; ii++)
                    {
                        deviceRequirements.Add(scheme.deviceRequirements[ii]);
                    }
                }
            }

            var inputControlScheme = new InputControlScheme(
                mlAgentsControlSchemeName,
                deviceRequirements);

            return inputControlScheme;
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="defaultMap"></param>
        /// <param name="layoutName"></param>
        /// <returns></returns>
        internal static void RegisterLayoutBuilder(InputActionMap defaultMap, string layoutName)
        {
            if (InputSystem.LoadLayout(layoutName) == null)
            {
                InputSystem.RegisterLayoutBuilder(() =>
                {
                    // TODO does this need to change based on the action map we use?
                    var builder = new InputControlLayout.Builder()
                        .WithName(layoutName)
                        .WithFormat(mlAgentsLayoutFormat);
                    for (var i = 0; i < defaultMap.actions.Count; i++)
                    {
                        var action = defaultMap.actions[i];
                        builder.AddControl(action.name)
                            .WithLayout(action.expectedControlType);
                    }
                    return builder.Build();
                }, layoutName);
            }
        }

        internal void FindNeededComponents()
        {
            if (m_InputAsset == null)
            {
                var assetProvider = GetComponent<IInputActionAssetProvider>();
                Assert.IsNotNull(assetProvider);
                (m_InputAsset, m_AssetCollection) = assetProvider.GetInputActionAsset();
                Assert.IsNotNull(m_InputAsset, "An InputActionAsset could not be found on IInputActionAssetProvider or PlayerInput.");
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
                m_BehaviorParameters.OnPolicyUpdated += UpdateDeviceBinding;
                m_LayoutName = mlAgentsLayoutName + m_BehaviorParameters.BehaviorName;
            }
        }

        internal void CleanupActionAsset()
        {
            InputSystem.RemoveLayout(mlAgentsLayoutName);
            if (!ReferenceEquals(m_Device, null))
            {
                InputSystem.RemoveDevice(m_Device);
            }

            if (!ReferenceEquals(m_InputAsset, null)
                && m_InputAsset.FindControlSchemeIndex(mlAgentsControlSchemeName) != -1)
            {
                m_InputAsset.RemoveControlScheme(mlAgentsControlSchemeName);
            }

            if (m_Actuators != null)
            {
                Array.Clear(m_Actuators, 0, m_Actuators.Length);
            }

            if (!ReferenceEquals(m_BehaviorParameters, null))
            {
                m_BehaviorParameters.OnPolicyUpdated -= UpdateDeviceBinding;
            }

            m_InputAsset = null;
            m_PlayerInput = null;
            m_BehaviorParameters = null;
            m_Device = null;
        }

        int m_ActuatorsWrittenToEvent;
        NativeArray<byte> m_InputBufferForFrame;
        InputEventPtr m_InputEventPtrForFrame;
        public InputEventPtr GetEventForFrame()
        {
#if UNITY_EDITOR
            if (!EditorApplication.isPlaying)
            {
                return new InputEventPtr();
            }
#endif
            if (m_ActuatorsWrittenToEvent % m_Actuators.Length == 0 || !m_InputEventPtrForFrame.valid)
            {
                m_ActuatorsWrittenToEvent = 0;
                m_InputEventPtrForFrame = new InputEventPtr();
                m_InputBufferForFrame = StateEvent.From(m_Device, out m_InputEventPtrForFrame);
            }

            return m_InputEventPtrForFrame;
        }

        public void EventProcessedInFrame()
        {
#if UNITY_EDITOR
            if (!EditorApplication.isPlaying)
            {
                return;
            }
#endif
            m_ActuatorsWrittenToEvent++;
            if (m_ActuatorsWrittenToEvent == m_Actuators.Length && m_InputEventPtrForFrame.valid)
            {
                InputSystem.QueueEvent(m_InputEventPtrForFrame);
                m_InputBufferForFrame.Dispose();
            }
        }
    }
}
#endif // MLA_INPUT_SYSTEM
