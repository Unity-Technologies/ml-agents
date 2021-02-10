using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

//[ExecuteAlways]
public class FPSAgentInput : MonoBehaviour
{
    public bool DisableInput = false;
    private FPSPlayerInputActions inputActions;
    private FPSPlayerInputActions.PlayerActions actionMap;
    private Gamepad gamepad;

    public Vector2 moveInput;

    public bool shootInput;
    public bool jumpInput;
    public bool dashInput;
    public Vector2 rotateInput;
    public bool shieldInput;
    // public Vector2 rotateVector2;
    public Camera Cam;
    // Start is called before the first frame update
    void Awake()
    {
        inputActions = new FPSPlayerInputActions();
        actionMap = inputActions.Player;
        Cursor.lockState = CursorLockMode.Locked;
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

    public bool shootPressed;
    // Update is called once per frame
    void FixedUpdate()
    {
        //        Vector2 move = gamepad.leftStick.ReadValue();

        if (DisableInput)
        {
            return;
        }
        moveInput = actionMap.Walk.ReadValue<Vector2>();
        shootInput = actionMap.Shoot.ReadValue<float>() > 0;
        shootPressed = actionMap.Shoot.triggered;
        //        shootInput = gamepad.rightTrigger.isPressed;
        shieldInput = actionMap.Shield.ReadValue<float>() > 0;
        //        rotateInput = actionMap.RotateBody.ReadValue<Vector2>();
        //        rotateInput = actionMap.Rotate.ReadValue<float>() * .1f;
        // rotateInput = actionMap.Rotate.ReadValue<float>();
        // rotateVector2 = actionMap.Rotate.ReadValue<Vector2>();
        rotateInput = actionMap.Rotate.ReadValue<Vector2>();
        // rotateInput = rotateVector2.x;
        //        rotateInput = actionMap.RotateBody.ReadValue<Vector2>();
        //        jumpInput = actionMap.Jump.ReadValue<float>() > 0;
        //        jumpInput = actionMap.Jump.triggered;
        if (actionMap.Dash.triggered)
        {
            dashInput = true;
        }
        if (actionMap.Jump.triggered)
        {
            jumpInput = true;
        }
    }
}
