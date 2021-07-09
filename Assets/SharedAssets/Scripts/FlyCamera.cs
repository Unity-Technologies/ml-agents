using UnityEngine;

namespace Unity.MLAgentsExamples
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

        Vector3
            m_LastMouse =
            new Vector3(255, 255,
                255);     // kind of in the middle of the screen, rather than at the top (play)

        float m_TotalRun = 1.0f;

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
                m_LastMouse = Input.mousePosition; // $CTK reset when we begin
            }

            if (!rotateOnlyIfMousedown ||
                (rotateOnlyIfMousedown && Input.GetMouseButton(1)))
            {
                m_LastMouse = Input.mousePosition - m_LastMouse;
                m_LastMouse = new Vector3(-m_LastMouse.y * camSens, m_LastMouse.x * camSens, 0);
                m_LastMouse = new Vector3(transform.eulerAngles.x + m_LastMouse.x,
                    transform.eulerAngles.y + m_LastMouse.y, 0);
                transform.eulerAngles = m_LastMouse;
                m_LastMouse = Input.mousePosition;
                // Mouse  camera angle done.
            }

            // Keyboard commands
            var p = GetBaseInput();
            if (Input.GetKey(KeyCode.LeftShift))
            {
                m_TotalRun += Time.deltaTime;
                p = shiftAdd * m_TotalRun * p;
                p.x = Mathf.Clamp(p.x, -maxShift, maxShift);
                p.y = Mathf.Clamp(p.y, -maxShift, maxShift);
                p.z = Mathf.Clamp(p.z, -maxShift, maxShift);
            }
            else
            {
                m_TotalRun = Mathf.Clamp(m_TotalRun * 0.5f, 1f, 1000f);
                p = p * mainSpeed;
            }

            p = p * Time.deltaTime;
            var newPosition = transform.position;
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

        Vector3 GetBaseInput()
        {
            // returns the basic values, if it's 0 than it's not active.
            var pVelocity = new Vector3();
            if (Input.GetKey(KeyCode.W))
            {
                pVelocity += new Vector3(0, 0, 1);
            }

            if (Input.GetKey(KeyCode.S))
            {
                pVelocity += new Vector3(0, 0, -1);
            }

            if (Input.GetKey(KeyCode.A))
            {
                pVelocity += new Vector3(-1, 0, 0);
            }

            if (Input.GetKey(KeyCode.D))
            {
                pVelocity += new Vector3(1, 0, 0);
            }

            return pVelocity;
        }
    }
}
