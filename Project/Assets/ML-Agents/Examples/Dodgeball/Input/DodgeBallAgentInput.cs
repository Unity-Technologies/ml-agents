using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

//[ExecuteAlways]
public class DodgeBallAgentInput : MonoBehaviour, IInputActionAssetProvider
{
    public bool DisableInput = false;
    public DodgeBallInputActions inputActions;
    private Gamepad gamepad;

    public Vector2 moveInput;

    public float throwInput;
    public bool jumpInput;
    public bool dashInput;
    public Vector2 rotateInput;
    public bool shieldInput;
    // public Vector2 rotateVector2;
    public Camera Cam;
    // Start is called before the first frame update
    void Awake()
    {
        LazyInitializeInput();
        Cursor.lockState = CursorLockMode.Locked;
    }

    void LazyInitializeInput()
    {
        if (ReferenceEquals(inputActions, null))
        {
            inputActions = new DodgeBallInputActions();
        }
    }

    void OnEnable()
    {
        gamepad = Gamepad.current;
        inputActions.Enable();
    }

    private void OnDisable()
    {
        inputActions.Disable();
    }

    public bool CheckIfInputSinceLastFrame(ref bool input)
    {
        if (input)
        {
            input = false;
            return true;
        }
        return false;
    }
    //    public bool JumpCheck(ref bool input)
    //    {
    //        if (jumped)
    //        {
    //            jumped = false;
    //            return true;
    //        }
    //        return false;
    //    }

    void Update()
    {
        if (inputActions.UI.Restart.triggered)
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
    }

    public bool throwPressed;
    // Update is called once per frame
    void FixedUpdate()
    {
        //        Vector2 move = gamepad.leftStick.ReadValue();

        if (DisableInput)
        {
            return;
        }
        moveInput = inputActions.Player.Walk.ReadValue<Vector2>();
        throwInput = inputActions.Player.Throw.ReadValue<float>();
        // throwPressed = actionMap.Throw.phase == InputActionPhase.Started;
        // throwPressed = actionMap.Throw.triggered;
        //        shootInput = gamepad.rightTrigger.isPressed;
        shieldInput = inputActions.Player.Shield.ReadValue<float>() > 0;
        //        rotateInput = actionMap.RotateBody.ReadValue<Vector2>();
        //        rotateInput = actionMap.Rotate.ReadValue<float>() * .1f;
        // rotateInput = actionMap.Rotate.ReadValue<float>();
        // rotateVector2 = actionMap.Rotate.ReadValue<Vector2>();
        rotateInput = inputActions.Player.Rotate.ReadValue<Vector2>();
        // rotateInput = rotateVector2.x;
        //        rotateInput = actionMap.RotateBody.ReadValue<Vector2>();
        //        jumpInput = actionMap.Jump.ReadValue<float>() > 0;
        //        jumpInput = actionMap.Jump.triggered;
        // if (actionMap.Throw.phase == InputActionPhase.Started)
        if (inputActions.Player.Throw.triggered)
        {
            throwPressed = true;
        }
        if (inputActions.Player.Dash.triggered)
        {
            dashInput = true;
        }
        if (inputActions.Player.Jump.triggered)
        {
            jumpInput = true;
        }
    }

    public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
    {
        LazyInitializeInput();
        return (inputActions.asset, inputActions);
    }
}
