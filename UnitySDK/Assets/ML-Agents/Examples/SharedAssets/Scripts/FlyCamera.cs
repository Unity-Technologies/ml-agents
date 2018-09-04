using UnityEngine;
using System.Collections;

namespace MLAgents
{

    public class FlyCamera : MonoBehaviour
    {

        /*
        wasd : basic movement
        shift : Makes camera accelerate
        space : Moves camera on X and Z axis only.  So camera doesn't gain any height*/


        public float mainSpeed = 100.0f; // regular speed
        public float shiftAdd = 250.0f; // multiplied by how long shift is held.  Basically running
        public float maxShift = 1000.0f; // Maximum speed when holdin gshift
        public float camSens = 0.25f; // How sensitive it with mouse
        public bool rotateOnlyIfMousedown = true;
        public bool movementStaysFlat = true;

        private Vector3
            lastMouse =
                new Vector3(255, 255,
                    255); // kind of in the middle of the screen, rather than at the top (play)

        private float totalRun = 1.0f;

        void Awake()
        {
            Debug.Log("FlyCamera Awake() - RESETTING CAMERA POSITION"); // nop?
            // nop:
            // transform.position.Set(0,8,-32);
            // transform.rotation.Set(15,0,0,1);
            transform.position = new Vector3(0, 8, -32);
            transform.rotation = Quaternion.Euler(25, 0, 0);
        }


        void Update()
        {

            if (Input.GetMouseButtonDown(1))
            {
                lastMouse = Input.mousePosition; // $CTK reset when we begin
            }

            if (!rotateOnlyIfMousedown ||
                (rotateOnlyIfMousedown && Input.GetMouseButton(1)))
            {
                lastMouse = Input.mousePosition - lastMouse;
                lastMouse = new Vector3(-lastMouse.y * camSens, lastMouse.x * camSens, 0);
                lastMouse = new Vector3(transform.eulerAngles.x + lastMouse.x,
                    transform.eulerAngles.y + lastMouse.y, 0);
                transform.eulerAngles = lastMouse;
                lastMouse = Input.mousePosition;
                // Mouse  camera angle done.  
            }

            // Keyboard commands
            Vector3 p = GetBaseInput();
            if (Input.GetKey(KeyCode.LeftShift))
            {
                totalRun += Time.deltaTime;
                p = p * totalRun * shiftAdd;
                p.x = Mathf.Clamp(p.x, -maxShift, maxShift);
                p.y = Mathf.Clamp(p.y, -maxShift, maxShift);
                p.z = Mathf.Clamp(p.z, -maxShift, maxShift);
            }
            else
            {
                totalRun = Mathf.Clamp(totalRun * 0.5f, 1f, 1000f);
                p = p * mainSpeed;
            }

            p = p * Time.deltaTime;
            Vector3 newPosition = transform.position;
            if (Input.GetKey(KeyCode.Space)
                || (movementStaysFlat && !(rotateOnlyIfMousedown && Input.GetMouseButton(1))))
            {
                // If player wants to move on X and Z axis only
                transform.Translate(p);
                newPosition.x = transform.position.x;
                newPosition.z = transform.position.z;
                transform.position = newPosition;
            }
            else
            {
                transform.Translate(p);
            }

        }

        private Vector3 GetBaseInput()
        {
            // returns the basic values, if it's 0 than it's not active.
            Vector3 p_Velocity = new Vector3();
            if (Input.GetKey(KeyCode.W))
            {
                p_Velocity += new Vector3(0, 0, 1);
            }

            if (Input.GetKey(KeyCode.S))
            {
                p_Velocity += new Vector3(0, 0, -1);
            }

            if (Input.GetKey(KeyCode.A))
            {
                p_Velocity += new Vector3(-1, 0, 0);
            }

            if (Input.GetKey(KeyCode.D))
            {
                p_Velocity += new Vector3(1, 0, 0);
            }

            return p_Velocity;
        }
    }
}
