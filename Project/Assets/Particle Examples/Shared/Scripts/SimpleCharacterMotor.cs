using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(CharacterController))]
public class SimpleCharacterMotor : MonoBehaviour
{
    public CursorLockMode cursorLockMode = CursorLockMode.Locked;
    public bool cursorVisible = false;
    [Header("Movement")]
    public float walkSpeed = 2;
    public float runSpeed = 4;
    public float gravity = 9.8f;
    [Space]
    [Header("Look")]
    public Transform cameraPivot;
    public float lookSpeed = 45;
    public bool invertY = true;
    [Space]
    [Header("Smoothing")]
    public float movementAcceleration = 1;

    CharacterController controller;
    Vector3 movement, finalMovement;
    float speed;
    Quaternion targetRotation, targetPivotRotation;


    void Awake()
    {
        controller = GetComponent<CharacterController>();
        Cursor.lockState = cursorLockMode;
        Cursor.visible = cursorVisible;
        targetRotation = targetPivotRotation = Quaternion.identity;
    }

    void Update()
    {
        UpdateTranslation();
        UpdateLookRotation();
    }

    void UpdateLookRotation()
    {
        var x = Input.GetAxis("Mouse Y");
        var y = Input.GetAxis("Mouse X");

        x *= invertY ? -1 : 1;
        targetRotation = transform.localRotation * Quaternion.AngleAxis(y * lookSpeed * Time.deltaTime, Vector3.up);
        targetPivotRotation = cameraPivot.localRotation * Quaternion.AngleAxis(x * lookSpeed * Time.deltaTime, Vector3.right);

        transform.localRotation = targetRotation;
        cameraPivot.localRotation = targetPivotRotation;
    }

    void UpdateTranslation()
    {
        if (controller.isGrounded)
        {
            var x = Input.GetAxis("Horizontal");
            var z = Input.GetAxis("Vertical");
            var run = Input.GetKey(KeyCode.LeftShift);

            var translation = new Vector3(x, 0, z);
            speed = run ? runSpeed : walkSpeed;
            movement = transform.TransformDirection(translation * speed);
        }
        else
        {
            movement.y -= gravity * Time.deltaTime;
        }
        finalMovement = Vector3.Lerp(finalMovement, movement, Time.deltaTime * movementAcceleration);
        controller.Move(finalMovement * Time.deltaTime);
    }
}
