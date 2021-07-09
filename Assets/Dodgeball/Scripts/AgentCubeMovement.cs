//Standardized movement controller for the Agent Cube
using Unity.MLAgents;
using UnityEngine;

namespace MLAgents
{
    public class AgentCubeMovement : MonoBehaviour
    {

        //ONLY ALLOW SCRIPTED MOVEMENT VIA ML-AGENTS OR OTHER HEURISTIC SCRIPTS
        [Header("INPUT")]
        public bool allowHumanInputAndDisableAgentHeuristicInput = true;

        [Header("RIGIDBODY")] public float maxAngularVel = 50;
        [Header("RUNNING")] public ForceMode runningForceMode = ForceMode.Impulse;
        //speed agent can run if grounded
        public float agentRunSpeed = 10;
        public float agentTerminalVel = 20;
        //speed agent can run if not grounded
        public float agentRunInAirSpeed = 7f;

        [Header("DASH")]
        public float dashBoostForce = 20f;
        public ForceMode dashForceMode = ForceMode.Impulse;
        public bool dashPressed;
        public float dashCoolDownDuration = .2f;
        public float dashCoolDownTimer;

        [Header("IDLE")]
        //coefficient used to dampen velocity when idle
        //the purpose of this is to fine tune agent drag
        //...and prevent the agent sliding around while grounded
        //0 means it will instantly stop when grounded
        //1 means no drag will be applied
        public float agentIdleDragVelCoeff = .9f;

        public enum RotationAxes { MouseXAndY = 0, MouseX = 1, MouseY = 2 };

        [Header("BODY ROTATION")]
        public float MouseSensitivity = 1;
        // public float mouseSmoothing = 0.5f;
        public float MouseSmoothTime = 0.05f;
        private float m_Yaw;
        private float m_SmoothYaw;
        private float m_YawSmoothV;
        Quaternion originalRotation;

        [Header("FALLING FORCE")]
        //force applied to agent while falling
        public float agentFallingSpeed = 50f;

        [Header("ANIMATE MESH")] public bool AnimateBodyMesh;
        public AnimationCurve walkingBounceCurve;
        public float walkingAnimScale = 1;
        public Transform bodyMesh;
        private float m_animateBodyMeshCurveTimer;

        private Rigidbody rb;
        public AgentCubeGroundCheck groundCheck;
        private float inputH;
        private float inputV;
        DodgeBallAgentInput m_Input;

        private DodgeBallAgent m_Agent;
        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            groundCheck = GetComponent<AgentCubeGroundCheck>();
            m_Agent = GetComponent<DodgeBallAgent>();
            rb.maxAngularVelocity = maxAngularVel;
            originalRotation = transform.localRotation;
            var envParameters = Academy.Instance.EnvironmentParameters;
            m_Input = GetComponent<DodgeBallAgentInput>();
        }

        public static float ClampAngle(float angle, float min, float max)
        {
            if (angle < -360F)
                angle += 360F;
            if (angle > 360F)
                angle -= 360F;
            return Mathf.Clamp(angle, min, max);
        }

        public void Look(float xRot = 0)
        {
            m_Yaw += xRot * MouseSensitivity;
            float smoothYawOld = m_SmoothYaw;
            m_SmoothYaw = Mathf.SmoothDampAngle(m_SmoothYaw, m_Yaw, ref m_YawSmoothV, MouseSmoothTime);
            rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(Mathf.DeltaAngle(smoothYawOld, m_SmoothYaw), transform.up));
        }

        void FixedUpdate()
        {
            dashCoolDownTimer += Time.fixedDeltaTime;

            if (groundCheck && !groundCheck.isGrounded)
            {
                AddFallingForce(rb);
            }

            if (m_Agent)
            {
                //this disables the heuristic input collection
                m_Agent.disableInputCollectionInHeuristicCallback = allowHumanInputAndDisableAgentHeuristicInput;
            }
            if (!allowHumanInputAndDisableAgentHeuristicInput || m_Agent.Stunned)
            {
                return;
            }

            float rotate = 0;
            if (!ReferenceEquals(null, m_Input))
            {
                rotate = m_Input.rotateInput;
                inputH = m_Input.moveInput.x;
                inputV = m_Input.moveInput.y;
            }
            var movDir = transform.TransformDirection(new Vector3(inputH, 0, inputV));
            RunOnGround(movDir);
            Look(rotate);

            if (m_Input.CheckIfInputSinceLastFrame(ref m_Input.m_dashPressed))
            {
                Dash(rb.transform.TransformDirection(new Vector3(inputH, 0, inputV)));
            }
            if (m_Agent && m_Input.CheckIfInputSinceLastFrame(ref m_Input.m_throwPressed))
            {
                m_Agent.ThrowTheBall();
            }
        }

        public void Dash(Vector3 dir)
        {
            if (dir != Vector3.zero && dashCoolDownTimer > dashCoolDownDuration)
            {
                rb.velocity = Vector3.zero;
                rb.AddForce(dir.normalized * dashBoostForce, dashForceMode);
                dashCoolDownTimer = 0;
            }
        }

        public void RotateTowards(Vector3 dir, float maxRotationRate = 1)
        {
            if (dir != Vector3.zero)
            {
                var rot = Quaternion.LookRotation(dir);
                var smoothedRot = Quaternion.RotateTowards(rb.rotation, rot, maxRotationRate * Time.deltaTime);
                rb.MoveRotation(smoothedRot);
            }
        }

        public void RunOnGround(Vector3 dir)
        {

            //ADD FORCE
            var vel = rb.velocity.magnitude;
            float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentTerminalVel);
            rb.AddForce(dir * adjustedSpeed, runningForceMode);

            //ANIMATE MESH
            if (dir == Vector3.zero)
            {
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero;
                }
            }
            else
            {
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero +
                                             Vector3.up * walkingAnimScale * walkingBounceCurve.Evaluate(
                                                 m_animateBodyMeshCurveTimer);
                    m_animateBodyMeshCurveTimer += Time.fixedDeltaTime;
                }
            }
        }

        public void RunInAir(Rigidbody rb, Vector3 dir)
        {
            var vel = rb.velocity.magnitude;
            float adjustedSpeed = Mathf.Clamp(agentRunInAirSpeed - vel, 0, agentTerminalVel);
            rb.AddForce(dir.normalized * adjustedSpeed,
                runningForceMode);
        }

        public void AddIdleDrag(Rigidbody rb)
        {
            rb.velocity *= agentIdleDragVelCoeff;
        }

        public void AddFallingForce(Rigidbody rb)
        {
            rb.AddForce(
                Vector3.down * agentFallingSpeed, ForceMode.Acceleration);
        }
    }
}
