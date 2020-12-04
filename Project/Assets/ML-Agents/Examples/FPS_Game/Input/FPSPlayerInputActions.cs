// GENERATED AUTOMATICALLY FROM 'Assets/ML-Agents/Examples/FPS_Game/Input/FPSPlayerInputActions.inputactions'

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Utilities;

public class @FPSPlayerInputActions : IInputActionCollection, IDisposable
{
    public InputActionAsset asset { get; }
    public @FPSPlayerInputActions()
    {
        asset = InputActionAsset.FromJson(@"{
    ""name"": ""FPSPlayerInputActions"",
    ""maps"": [
        {
            ""name"": ""Player Action Map"",
            ""id"": ""bb797917-97ca-47e1-b3c9-0572380e9376"",
            ""actions"": [
                {
                    ""name"": ""Walk"",
                    ""type"": ""Value"",
                    ""id"": ""2f9ecc77-85d9-4189-8faf-18dcf905d2d4"",
                    ""expectedControlType"": ""Vector2"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""DashLeft"",
                    ""type"": ""Button"",
                    ""id"": ""1d14cda7-c4b3-4bb4-9ebb-f197aa5ed183"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""DashRight"",
                    ""type"": ""Button"",
                    ""id"": ""96db7975-8803-410a-863b-5f25b31019f4"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""Jump"",
                    ""type"": ""Button"",
                    ""id"": ""d2faa0a0-e027-4a5d-a155-870bb0ceaf7c"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""Shoot"",
                    ""type"": ""Button"",
                    ""id"": ""cbcb2a57-a474-46a7-b133-cf144f6de321"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""Shield"",
                    ""type"": ""Button"",
                    ""id"": ""ec97db61-8659-49c6-afaa-20b82a98e72b"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""RotateBody"",
                    ""type"": ""Value"",
                    ""id"": ""379999a6-e908-4242-ae2f-384e38bcb7cb"",
                    ""expectedControlType"": ""Vector2"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""Dash"",
                    ""type"": ""Button"",
                    ""id"": ""49d5245a-a350-4f94-b4c8-cf578d61000b"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": """"
                },
                {
                    ""name"": ""Rotate"",
                    ""type"": ""Value"",
                    ""id"": ""b309ca48-95aa-4032-bd7d-46dcaeb138d0"",
                    ""expectedControlType"": ""Axis"",
                    ""processors"": """",
                    ""interactions"": """"
                }
            ],
            ""bindings"": [
                {
                    ""name"": ""2D Vector"",
                    ""id"": ""f7a4010c-91c6-422f-91e7-4b14cbdb214a"",
                    ""path"": ""2DVector(mode=2)"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": true,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""up"",
                    ""id"": ""0067bacb-5284-4358-8fbf-cbbd17bd4884"",
                    ""path"": ""<Keyboard>/w"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""down"",
                    ""id"": ""5e100fc2-94e3-4f73-8020-b6cbeed41b97"",
                    ""path"": ""<Keyboard>/s"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""left"",
                    ""id"": ""5eb9d69e-8df8-443c-93f4-447eeb52b75f"",
                    ""path"": ""<Keyboard>/a"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""right"",
                    ""id"": ""3b4de59c-d93c-4072-b11e-711ea8534593"",
                    ""path"": ""<Keyboard>/d"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": """",
                    ""id"": ""6fdb596c-c0bb-4f37-b409-c235e9612acf"",
                    ""path"": ""<Gamepad>/leftStick"",
                    ""interactions"": """",
                    ""processors"": ""StickDeadzone"",
                    ""groups"": """",
                    ""action"": ""Walk"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""14009289-6acb-4978-8838-01648248f103"",
                    ""path"": ""<Gamepad>/leftShoulder"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""DashLeft"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""28ddf0bd-ea22-4069-ba60-ca41d4742bf9"",
                    ""path"": ""<Gamepad>/buttonSouth"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Jump"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""6f4a5988-5e01-4c66-a3ab-f4b5e47bd21f"",
                    ""path"": ""<Keyboard>/j"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Jump"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""fe641861-0da7-43ba-8bd6-e95a17232a77"",
                    ""path"": ""<Gamepad>/rightTrigger"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Shoot"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""2ae82452-d746-4a28-86f4-43c469c78eff"",
                    ""path"": ""<Keyboard>/k"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Shoot"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""2b94ae52-9bb0-478e-8e4c-bf6c747a5c7d"",
                    ""path"": ""<Gamepad>/leftTrigger"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Shield"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""75b3d327-54da-4b56-a39e-052b1c173b56"",
                    ""path"": ""<Keyboard>/i"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Shield"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""4df83c50-5bb2-497d-a713-237e1c5dc25a"",
                    ""path"": ""<Gamepad>/rightShoulder"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""DashRight"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""0c0d917f-0af9-4575-b970-f441fd97db20"",
                    ""path"": ""<Gamepad>/rightStick"",
                    ""interactions"": """",
                    ""processors"": ""StickDeadzone(min=0.15,max=0.925)"",
                    ""groups"": """",
                    ""action"": ""RotateBody"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""319796b1-6071-46f0-81dc-58b6bdb7d86a"",
                    ""path"": ""<Gamepad>/buttonEast"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Dash"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""ae8dc5a5-930d-412e-a196-627e63bfcd0c"",
                    ""path"": ""<Keyboard>/l"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Dash"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""1D Axis"",
                    ""id"": ""3e2a8f42-5d52-42d2-bd25-2934ab0f8c53"",
                    ""path"": ""1DAxis"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": true,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""negative"",
                    ""id"": ""561ac5e2-b709-4ae7-8151-1777260a4798"",
                    ""path"": ""<Keyboard>/q"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""positive"",
                    ""id"": ""23905e8a-1906-4f34-8e8f-5e6fa596e27e"",
                    ""path"": ""<Keyboard>/e"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""1D Axis"",
                    ""id"": ""a788bd05-d81e-47f4-a14c-5cb071e21c96"",
                    ""path"": ""1DAxis"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": true,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""negative"",
                    ""id"": ""9844ff0b-6a74-43bb-a969-9f59c74212b9"",
                    ""path"": ""<Gamepad>/rightStick/left"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""positive"",
                    ""id"": ""b81e038b-8786-4bbe-bb1d-aba1d54112d1"",
                    ""path"": ""<Gamepad>/rightStick/right"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                }
            ]
        }
    ],
    ""controlSchemes"": []
}");
        // Player Action Map
        m_PlayerActionMap = asset.FindActionMap("Player Action Map", throwIfNotFound: true);
        m_PlayerActionMap_Walk = m_PlayerActionMap.FindAction("Walk", throwIfNotFound: true);
        m_PlayerActionMap_DashLeft = m_PlayerActionMap.FindAction("DashLeft", throwIfNotFound: true);
        m_PlayerActionMap_DashRight = m_PlayerActionMap.FindAction("DashRight", throwIfNotFound: true);
        m_PlayerActionMap_Jump = m_PlayerActionMap.FindAction("Jump", throwIfNotFound: true);
        m_PlayerActionMap_Shoot = m_PlayerActionMap.FindAction("Shoot", throwIfNotFound: true);
        m_PlayerActionMap_Shield = m_PlayerActionMap.FindAction("Shield", throwIfNotFound: true);
        m_PlayerActionMap_RotateBody = m_PlayerActionMap.FindAction("RotateBody", throwIfNotFound: true);
        m_PlayerActionMap_Dash = m_PlayerActionMap.FindAction("Dash", throwIfNotFound: true);
        m_PlayerActionMap_Rotate = m_PlayerActionMap.FindAction("Rotate", throwIfNotFound: true);
    }

    public void Dispose()
    {
        UnityEngine.Object.Destroy(asset);
    }

    public InputBinding? bindingMask
    {
        get => asset.bindingMask;
        set => asset.bindingMask = value;
    }

    public ReadOnlyArray<InputDevice>? devices
    {
        get => asset.devices;
        set => asset.devices = value;
    }

    public ReadOnlyArray<InputControlScheme> controlSchemes => asset.controlSchemes;

    public bool Contains(InputAction action)
    {
        return asset.Contains(action);
    }

    public IEnumerator<InputAction> GetEnumerator()
    {
        return asset.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public void Enable()
    {
        asset.Enable();
    }

    public void Disable()
    {
        asset.Disable();
    }

    // Player Action Map
    private readonly InputActionMap m_PlayerActionMap;
    private IPlayerActionMapActions m_PlayerActionMapActionsCallbackInterface;
    private readonly InputAction m_PlayerActionMap_Walk;
    private readonly InputAction m_PlayerActionMap_DashLeft;
    private readonly InputAction m_PlayerActionMap_DashRight;
    private readonly InputAction m_PlayerActionMap_Jump;
    private readonly InputAction m_PlayerActionMap_Shoot;
    private readonly InputAction m_PlayerActionMap_Shield;
    private readonly InputAction m_PlayerActionMap_RotateBody;
    private readonly InputAction m_PlayerActionMap_Dash;
    private readonly InputAction m_PlayerActionMap_Rotate;
    public struct PlayerActionMapActions
    {
        private @FPSPlayerInputActions m_Wrapper;
        public PlayerActionMapActions(@FPSPlayerInputActions wrapper) { m_Wrapper = wrapper; }
        public InputAction @Walk => m_Wrapper.m_PlayerActionMap_Walk;
        public InputAction @DashLeft => m_Wrapper.m_PlayerActionMap_DashLeft;
        public InputAction @DashRight => m_Wrapper.m_PlayerActionMap_DashRight;
        public InputAction @Jump => m_Wrapper.m_PlayerActionMap_Jump;
        public InputAction @Shoot => m_Wrapper.m_PlayerActionMap_Shoot;
        public InputAction @Shield => m_Wrapper.m_PlayerActionMap_Shield;
        public InputAction @RotateBody => m_Wrapper.m_PlayerActionMap_RotateBody;
        public InputAction @Dash => m_Wrapper.m_PlayerActionMap_Dash;
        public InputAction @Rotate => m_Wrapper.m_PlayerActionMap_Rotate;
        public InputActionMap Get() { return m_Wrapper.m_PlayerActionMap; }
        public void Enable() { Get().Enable(); }
        public void Disable() { Get().Disable(); }
        public bool enabled => Get().enabled;
        public static implicit operator InputActionMap(PlayerActionMapActions set) { return set.Get(); }
        public void SetCallbacks(IPlayerActionMapActions instance)
        {
            if (m_Wrapper.m_PlayerActionMapActionsCallbackInterface != null)
            {
                @Walk.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnWalk;
                @Walk.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnWalk;
                @Walk.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnWalk;
                @DashLeft.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashLeft;
                @DashLeft.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashLeft;
                @DashLeft.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashLeft;
                @DashRight.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashRight;
                @DashRight.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashRight;
                @DashRight.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDashRight;
                @Jump.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnJump;
                @Jump.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnJump;
                @Jump.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnJump;
                @Shoot.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShoot;
                @Shoot.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShoot;
                @Shoot.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShoot;
                @Shield.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShield;
                @Shield.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShield;
                @Shield.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnShield;
                @RotateBody.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotateBody;
                @RotateBody.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotateBody;
                @RotateBody.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotateBody;
                @Dash.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDash;
                @Dash.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDash;
                @Dash.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnDash;
                @Rotate.started -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotate;
                @Rotate.performed -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotate;
                @Rotate.canceled -= m_Wrapper.m_PlayerActionMapActionsCallbackInterface.OnRotate;
            }
            m_Wrapper.m_PlayerActionMapActionsCallbackInterface = instance;
            if (instance != null)
            {
                @Walk.started += instance.OnWalk;
                @Walk.performed += instance.OnWalk;
                @Walk.canceled += instance.OnWalk;
                @DashLeft.started += instance.OnDashLeft;
                @DashLeft.performed += instance.OnDashLeft;
                @DashLeft.canceled += instance.OnDashLeft;
                @DashRight.started += instance.OnDashRight;
                @DashRight.performed += instance.OnDashRight;
                @DashRight.canceled += instance.OnDashRight;
                @Jump.started += instance.OnJump;
                @Jump.performed += instance.OnJump;
                @Jump.canceled += instance.OnJump;
                @Shoot.started += instance.OnShoot;
                @Shoot.performed += instance.OnShoot;
                @Shoot.canceled += instance.OnShoot;
                @Shield.started += instance.OnShield;
                @Shield.performed += instance.OnShield;
                @Shield.canceled += instance.OnShield;
                @RotateBody.started += instance.OnRotateBody;
                @RotateBody.performed += instance.OnRotateBody;
                @RotateBody.canceled += instance.OnRotateBody;
                @Dash.started += instance.OnDash;
                @Dash.performed += instance.OnDash;
                @Dash.canceled += instance.OnDash;
                @Rotate.started += instance.OnRotate;
                @Rotate.performed += instance.OnRotate;
                @Rotate.canceled += instance.OnRotate;
            }
        }
    }
    public PlayerActionMapActions @PlayerActionMap => new PlayerActionMapActions(this);
    public interface IPlayerActionMapActions
    {
        void OnWalk(InputAction.CallbackContext context);
        void OnDashLeft(InputAction.CallbackContext context);
        void OnDashRight(InputAction.CallbackContext context);
        void OnJump(InputAction.CallbackContext context);
        void OnShoot(InputAction.CallbackContext context);
        void OnShield(InputAction.CallbackContext context);
        void OnRotateBody(InputAction.CallbackContext context);
        void OnDash(InputAction.CallbackContext context);
        void OnRotate(InputAction.CallbackContext context);
    }
}
