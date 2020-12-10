using UnityEngine;

namespace Cinemachine.Examples
{

    [AddComponentMenu("")] // Don't display in add component menu
    public class CharacterMovement2D : MonoBehaviour
    {
        public KeyCode sprintJoystick = KeyCode.JoystickButton2;
        public KeyCode jumpJoystick = KeyCode.JoystickButton0;
        public KeyCode sprintKeyboard = KeyCode.LeftShift;
        public KeyCode jumpKeyboard = KeyCode.Space;
        public float jumpVelocity = 7f;
        public float groundTolerance = 0.2f;
        public bool checkGroundForJump = true;

        private float speed = 0f;
        private bool isSprinting = false;
        private Animator anim;
        private Vector2 input;
        private float velocity;
        private bool headingleft = false;
        private Quaternion targetrot;
        private Rigidbody rigbody;

        // Use this for initialization
        void Start()
        {
            anim = GetComponent<Animator>();
            rigbody = GetComponent<Rigidbody>();
            targetrot = transform.rotation;
        }

        // Update is called once per frame
        void FixedUpdate()
        {
            input.x = Input.GetAxis("Horizontal");

            // Check if direction changes
            if ((input.x < 0f && !headingleft) || (input.x > 0f && headingleft))
            {
                if (input.x < 0f) targetrot = Quaternion.Euler(0, 270, 0);
                if (input.x > 0f) targetrot = Quaternion.Euler(0, 90, 0);
                headingleft = !headingleft;
            }
            // Rotate player if direction changes
            transform.rotation = Quaternion.Lerp(transform.rotation, targetrot, Time.deltaTime * 20f);

            // set speed to horizontal inputs
            speed = Mathf.Abs(input.x);
            speed = Mathf.SmoothDamp(anim.GetFloat("Speed"), speed, ref velocity, 0.1f);
            anim.SetFloat("Speed", speed);

            // set sprinting
            if ((Input.GetKeyDown(sprintJoystick) || Input.GetKeyDown(sprintKeyboard)) && input != Vector2.zero) isSprinting = true;
            if ((Input.GetKeyUp(sprintJoystick) || Input.GetKeyUp(sprintKeyboard)) || input == Vector2.zero) isSprinting = false;
            anim.SetBool("isSprinting", isSprinting);

            // Jump
            if ((Input.GetKeyDown(jumpJoystick) || Input.GetKeyDown(jumpKeyboard)) && isGrounded())
            {
                rigbody.velocity = new Vector3(input.x, jumpVelocity, 0f);
            }
        }

        public bool isGrounded()
        {
            if (checkGroundForJump)
                return Physics.Raycast(transform.position, Vector3.down, groundTolerance);
            else
                return true;
        }
    }

}
