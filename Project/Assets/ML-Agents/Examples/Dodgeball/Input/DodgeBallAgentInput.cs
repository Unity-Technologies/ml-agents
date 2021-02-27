using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

//[ExecuteAlways]
public class DodgeBallAgentInput : MonoBehaviour
{
    public bool DisableInput = false;
    private DodgeBallInputActions inputActions;
    private DodgeBallInputActions.PlayerActions actionMap;
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
        inputActions = new DodgeBallInputActions();
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

    public bool throwPressed;
    // Update is called once per frame
    void FixedUpdate()
    {
        //        Vector2 move = gamepad.leftStick.ReadValue();

        if (DisableInput)
        {
            return;
        }
        moveInput = actionMap.Walk.ReadValue<Vector2>();
        throwInput = actionMap.Throw.ReadValue<float>();
        // throwPressed = actionMap.Throw.phase == InputActionPhase.Started;
        // throwPressed = actionMap.Throw.triggered;
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
        // if (actionMap.Throw.phase == InputActionPhase.Started)
        if (actionMap.Throw.triggered)
        {
            throwPressed = true;
        }
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
