// GENERATED AUTOMATICALLY FROM 'Assets/ML-Agents/Examples/PushBlock/PushBlockActions.inputactions'

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Utilities;

public class @PushBlockActions : IInputActionCollection, IDisposable
{
    public InputActionAsset asset { get; }
    public @PushBlockActions()
    {
        asset = InputActionAsset.FromJson(@"{
    ""name"": ""PushBlockActions"",
    ""maps"": [
        {
            ""name"": ""Movement"",
            ""id"": ""03a2e5d4-ae81-47f1-a575-0779fb7da538"",
            ""actions"": [
                {
                    ""name"": ""Foward"",
                    ""type"": ""Button"",
                    ""id"": ""a1607d28-b552-4055-a69b-7c0ad843da3c"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": ""Press""
                },
                {
                    ""name"": ""Backward"",
                    ""type"": ""Button"",
                    ""id"": ""3fd5c676-66cb-4faf-8833-bd11c6bd1b69"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": ""Press""
                },
                {
                    ""name"": ""Rotate Right"",
                    ""type"": ""Button"",
                    ""id"": ""46be909c-9865-49bf-8ef3-6b82af491848"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": ""Press""
                },
                {
                    ""name"": ""Rotate Left"",
                    ""type"": ""Button"",
                    ""id"": ""2ea2090e-5adb-4254-b807-41735f3330be"",
                    ""expectedControlType"": ""Button"",
                    ""processors"": """",
                    ""interactions"": ""Press""
                },
                {
                    ""name"": ""New action"",
                    ""type"": ""Value"",
                    ""id"": ""ee15661e-d945-4392-887f-28aaabb5ef00"",
                    ""expectedControlType"": ""Vector2"",
                    ""processors"": """",
                    ""interactions"": ""Press""
                }
            ],
            ""bindings"": [
                {
                    ""name"": """",
                    ""id"": ""90b1a8aa-8cab-4a50-8f4b-168c5a6f25f0"",
                    ""path"": ""<Keyboard>/w"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Foward"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""63e89251-9c68-4c75-b4a4-9fed1faf376f"",
                    ""path"": ""<Keyboard>/s"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Backward"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""6a18f838-1813-4f00-af54-446ea6559823"",
                    ""path"": ""<Keyboard>/d"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate Right"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""140ddca0-c1ca-4fcf-8c36-2c482c387c81"",
                    ""path"": ""<Keyboard>/a"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""Rotate Left"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": """",
                    ""id"": ""bcbcfe67-7ca7-4fb9-aea2-dc8e1923c997"",
                    ""path"": """",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""2D Vector"",
                    ""id"": ""d6d82dac-291e-4757-946f-0aa54b80e474"",
                    ""path"": ""2DVector"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": true,
                    ""isPartOfComposite"": false
                },
                {
                    ""name"": ""up"",
                    ""id"": ""bd130402-ab1b-4d23-a574-b5e8dc590168"",
                    ""path"": ""*/{Forward}"",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""down"",
                    ""id"": ""196ca955-652e-4f3c-8448-c120edcecde8"",
                    ""path"": """",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""left"",
                    ""id"": ""ee7bfe04-3858-4f6f-be81-3a873cc0c325"",
                    ""path"": """",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                },
                {
                    ""name"": ""right"",
                    ""id"": ""5c590405-d6dc-4937-9a7d-5a559c649f75"",
                    ""path"": """",
                    ""interactions"": """",
                    ""processors"": """",
                    ""groups"": """",
                    ""action"": ""New action"",
                    ""isComposite"": false,
                    ""isPartOfComposite"": true
                }
            ]
        }
    ],
    ""controlSchemes"": []
}");
        // Movement
        m_Movement = asset.FindActionMap("Movement", throwIfNotFound: true);
        m_Movement_Foward = m_Movement.FindAction("Foward", throwIfNotFound: true);
        m_Movement_Backward = m_Movement.FindAction("Backward", throwIfNotFound: true);
        m_Movement_RotateRight = m_Movement.FindAction("Rotate Right", throwIfNotFound: true);
        m_Movement_RotateLeft = m_Movement.FindAction("Rotate Left", throwIfNotFound: true);
        m_Movement_Newaction = m_Movement.FindAction("New action", throwIfNotFound: true);
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

    // Movement
    private readonly InputActionMap m_Movement;
    private IMovementActions m_MovementActionsCallbackInterface;
    private readonly InputAction m_Movement_Foward;
    private readonly InputAction m_Movement_Backward;
    private readonly InputAction m_Movement_RotateRight;
    private readonly InputAction m_Movement_RotateLeft;
    private readonly InputAction m_Movement_Newaction;
    public struct MovementActions
    {
        private @PushBlockActions m_Wrapper;
        public MovementActions(@PushBlockActions wrapper) { m_Wrapper = wrapper; }
        public InputAction @Foward => m_Wrapper.m_Movement_Foward;
        public InputAction @Backward => m_Wrapper.m_Movement_Backward;
        public InputAction @RotateRight => m_Wrapper.m_Movement_RotateRight;
        public InputAction @RotateLeft => m_Wrapper.m_Movement_RotateLeft;
        public InputAction @Newaction => m_Wrapper.m_Movement_Newaction;
        public InputActionMap Get() { return m_Wrapper.m_Movement; }
        public void Enable() { Get().Enable(); }
        public void Disable() { Get().Disable(); }
        public bool enabled => Get().enabled;
        public static implicit operator InputActionMap(MovementActions set) { return set.Get(); }
        public void SetCallbacks(IMovementActions instance)
        {
            if (m_Wrapper.m_MovementActionsCallbackInterface != null)
            {
                @Foward.started -= m_Wrapper.m_MovementActionsCallbackInterface.OnFoward;
                @Foward.performed -= m_Wrapper.m_MovementActionsCallbackInterface.OnFoward;
                @Foward.canceled -= m_Wrapper.m_MovementActionsCallbackInterface.OnFoward;
                @Backward.started -= m_Wrapper.m_MovementActionsCallbackInterface.OnBackward;
                @Backward.performed -= m_Wrapper.m_MovementActionsCallbackInterface.OnBackward;
                @Backward.canceled -= m_Wrapper.m_MovementActionsCallbackInterface.OnBackward;
                @RotateRight.started -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateRight;
                @RotateRight.performed -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateRight;
                @RotateRight.canceled -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateRight;
                @RotateLeft.started -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateLeft;
                @RotateLeft.performed -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateLeft;
                @RotateLeft.canceled -= m_Wrapper.m_MovementActionsCallbackInterface.OnRotateLeft;
                @Newaction.started -= m_Wrapper.m_MovementActionsCallbackInterface.OnNewaction;
                @Newaction.performed -= m_Wrapper.m_MovementActionsCallbackInterface.OnNewaction;
                @Newaction.canceled -= m_Wrapper.m_MovementActionsCallbackInterface.OnNewaction;
            }
            m_Wrapper.m_MovementActionsCallbackInterface = instance;
            if (instance != null)
            {
                @Foward.started += instance.OnFoward;
                @Foward.performed += instance.OnFoward;
                @Foward.canceled += instance.OnFoward;
                @Backward.started += instance.OnBackward;
                @Backward.performed += instance.OnBackward;
                @Backward.canceled += instance.OnBackward;
                @RotateRight.started += instance.OnRotateRight;
                @RotateRight.performed += instance.OnRotateRight;
                @RotateRight.canceled += instance.OnRotateRight;
                @RotateLeft.started += instance.OnRotateLeft;
                @RotateLeft.performed += instance.OnRotateLeft;
                @RotateLeft.canceled += instance.OnRotateLeft;
                @Newaction.started += instance.OnNewaction;
                @Newaction.performed += instance.OnNewaction;
                @Newaction.canceled += instance.OnNewaction;
            }
        }
    }
    public MovementActions @Movement => new MovementActions(this);
    public interface IMovementActions
    {
        void OnFoward(InputAction.CallbackContext context);
        void OnBackward(InputAction.CallbackContext context);
        void OnRotateRight(InputAction.CallbackContext context);
        void OnRotateLeft(InputAction.CallbackContext context);
        void OnNewaction(InputAction.CallbackContext context);
    }
}
