using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class FPSAgentInput : MonoBehaviour
{
    private FPSPlayerInputActions inputActions;
    private FPSPlayerInputActions.PlayerActionMapActions actionMap;
    private Gamepad gamepad;

    public Vector2 moveInput;

    public bool shootInput;
    public bool jumpInput;
    public bool dashInput;
    public Vector2 rotateInput;
    public bool shieldInput;

    // Start is called before the first frame update
    void Awake()
    {
        inputActions = new FPSPlayerInputActions();
        actionMap = inputActions.PlayerActionMap;
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

    // Update is called once per frame
    void FixedUpdate()
    {
        //        Vector2 move = gamepad.leftStick.ReadValue();

        moveInput = actionMap.Walk.ReadValue<Vector2>();
        //        shootInput = actionMap.Shoot.ReadValue<float>() > 0;
        shootInput = gamepad.rightTrigger.isPressed;
        rotateInput = actionMap.RotateBody.ReadValue<Vector2>();
        //        jumpInput = actionMap.Jump.ReadValue<float>() > 0;
        //        jumpInput = actionMap.Jump.performed;
        jumpInput = gamepad.buttonSouth.isPressed;
        dashInput = gamepad.buttonWest.isPressed;
    }
}
