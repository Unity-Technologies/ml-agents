using UnityEngine;

// WASD to move, Space to sprint
public class CharacterMovementNoCamera : MonoBehaviour
{
    public Transform InvisibleCameraOrigin;

    public float StrafeSpeed = 0.1f;
    public float TurnSpeed = 3;
    public float Damping = 0.2f;
    public float VerticalRotMin = -80;
    public float VerticalRotMax = 80;
    public KeyCode sprintJoystick = KeyCode.JoystickButton2;
    public KeyCode sprintKeyboard = KeyCode.Space;

    private bool isSprinting;
    private Animator anim;
    private float currentStrafeSpeed;
    private Vector2 currentVelocity;

    void OnEnable()
    {
        anim = GetComponent<Animator>();
        currentVelocity = Vector2.zero;
        currentStrafeSpeed = 0;
        isSprinting = false;
    }

    void FixedUpdate()
    {
        var input = new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
        var speed = input.y;
        speed = Mathf.Clamp(speed, -1f, 1f);
        speed = Mathf.SmoothDamp(anim.GetFloat("Speed"), speed, ref currentVelocity.y, Damping);
        anim.SetFloat("Speed", speed);
        anim.SetFloat("Direction", speed);

        // set sprinting
        isSprinting = (Input.GetKey(sprintJoystick) || Input.GetKey(sprintKeyboard)) && speed > 0;
        anim.SetBool("isSprinting", isSprinting);

        // strafing
        currentStrafeSpeed = Mathf.SmoothDamp(
            currentStrafeSpeed, input.x * StrafeSpeed, ref currentVelocity.x, Damping);
        transform.position += transform.TransformDirection(Vector3.right) * currentStrafeSpeed;

        var rotInput = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
        var rot = transform.eulerAngles;
        rot.y += rotInput.x * TurnSpeed;
        transform.rotation = Quaternion.Euler(rot);

        if (InvisibleCameraOrigin != null)
        {
            rot = InvisibleCameraOrigin.localRotation.eulerAngles;
            rot.x -= rotInput.y * TurnSpeed;
            if (rot.x > 180)
                rot.x -= 360;
            rot.x = Mathf.Clamp(rot.x, VerticalRotMin, VerticalRotMax);
            InvisibleCameraOrigin.localRotation = Quaternion.Euler(rot);
        }
    }
}
