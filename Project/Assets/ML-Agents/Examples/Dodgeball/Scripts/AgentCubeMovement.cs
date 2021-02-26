//Standardized movement controller for the Agent Cube

using System;
using Unity.Mathematics;
using UnityEngine;

namespace MLAgents
{
    public class AgentCubeMovement : MonoBehaviour
    {

        //ONLY ALLOW SCRIPTED MOVEMENT VIA ML-AGENTS OR OTHER HEURISTIC SCRIPTS
        [Header("INPUT")]
        public bool allowHumanInput = true;
        // public DodgeBallAgentInput input;



        [Header("RIGIDBODY")] public float maxAngularVel = 50;
        [Header("RUNNING")] public ForceMode runningForceMode = ForceMode.Impulse;
        //speed agent can run if grounded
        public float agentRunSpeed = 10;
        public float agentTerminalVel = 20;
        //speed agent can run if not grounded
        public float agentRunInAirSpeed = 7f;

        [Header("STRAFE")]
        public float strafeSpeed = 10;
        public float strafeCoolDownDuration = .2f;
        public float strafeCoolDownTimer;
        public ForceMode strafeForceMode = ForceMode.Impulse;

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

        [Header("GROUND POUND")] public bool UseGroundPound;
        public ForceMode groundPoundForceMode = ForceMode.Impulse;
        public float groundPoundForce = 35f;

        [Header("SPIN ATTACK")]
        public float spinAttackSpeed = 20f;
        private bool spinAttack;

        public enum RotationAxes { MouseXAndY = 0, MouseX = 1, 	MouseY = 2 };

        [Header("BODY ROTATION")] public bool UseMouseRotation = true;
        public RotationAxes axes = RotationAxes.MouseXAndY;
        public float sensitivityX = 15F;
        public float sensitivityY = 15F;

        public float minimumX = -360F;
        public float maximumX = 360F;

        public float minimumY = -60F;
        public float maximumY = 60F;

        float rotationX = 0F;
        float rotationY = 0F;

        Quaternion originalRotation;

        [Header("BODY ROTATION")]
        public bool MatchCameraRotation;
        //body rotation speed
        public bool invertRotationIfWalkingBackwards = true;
        public float agentRotationSpeed = 35f;
        private Vector2 m_Rotation;

        [Header("JUMPING")] public bool canJump = true;
        //upward jump velocity magnitude
        public float agentJumpVelocity = 15f;

        [Header("FALLING FORCE")]
        //force applied to agent while falling
        public float agentFallingSpeed = 50f;

        [Header("ANIMATE MESH")] public bool AnimateBodyMesh;
        public AnimationCurve walkingBounceCurve;
        public float walkingAnimScale = 1;
        public Transform bodyMesh;
        private float m_animateBodyMeshCurveTimer;

        public Camera cam;
        private float lookDir;
        private Rigidbody rb;
        public AgentCubeGroundCheck groundCheck;
        private float inputH;
        private float inputV;
        private bool jump;
        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            groundCheck = GetComponent<AgentCubeGroundCheck>();
            rb.maxAngularVelocity = maxAngularVel;
            originalRotation = transform.localRotation;
        }

        void Update()
        {
            if (!allowHumanInput)
            {
                //FORWARD MOVEMENT
                inputV = Input.GetAxisRaw("Vertical");
                inputH = Input.GetAxisRaw("Horizontal");
                spinAttack = Input.GetKey(KeyCode.H);
                dashPressed = Input.GetKeyDown(KeyCode.K);
                jump = Input.GetKeyDown(KeyCode.Space);
                return;
            }
            var camForward = cam.transform.forward;
            camForward.y = 0;
            var camRight = cam.transform.right;

            if (canJump && Input.GetKeyDown(KeyCode.Space))
            {
                if (groundCheck)
                {
                    if (groundCheck.isGrounded)
                    {
                        Jump();
                    }
                    else if (UseGroundPound)
                    {
                        rb.AddForce(Vector3.down * groundPoundForce, groundPoundForceMode);
                    }
                }
            }



            if (dashPressed)
            {
                //                dashPressed = false;
                //                rb.AddForce(rb.transform.forward * dashBoostForce, dashForceMode);
                rb.AddTorque(rb.transform.right * dashBoostForce, dashForceMode);
                print("dashPressed");
            }
        }


        public void RotateBody(float rotateAxis, float forwardAxis)
        {
            // var walkingBackwardsCoeff = 1;
            // if (invertRotationIfWalkingBackwards && forwardAxis < 0)
            // {
            //     walkingBackwardsCoeff = -1;
            // }
            // var scaledRotateSpeed = agentRotationSpeed * Time.fixedDeltaTime;
            // var amount = Quaternion.Euler(0, rotateAxis * walkingBackwardsCoeff , 0);

            // var amount = Quaternion.Euler(0, rotateAxis * agentRotationSpeed * walkingBackwardsCoeff * Time.fixedDeltaTime, 0);
            var amount = Quaternion.Euler(0, rotateAxis * agentRotationSpeed * Time.fixedDeltaTime, 0);
            rb.rotation *= amount;
            // rb.MoveRotation(rb.rotation * amount);
        }

        public static float ClampAngle(float angle, float min, float max)
        {
            if (angle < -360F)
                angle += 360F;
            if (angle > 360F)
                angle -= 360F;
            return Mathf.Clamp(angle, min, max);
        }

        public void Look(float xRot = 0, float yRot = 0)
        {
                // Read the mouse input axis
                rotationX += xRot * sensitivityX;
                rotationY += yRot * sensitivityY;
                // rotationX += xRot * sensitivityX * Time.deltaTime;
                // rotationY += yRot * sensitivityY * Time.deltaTime;
                // rotationX += Input.GetAxis("Mouse X") * sensitivityX;
                // rotationY += Input.GetAxis("Mouse Y") * sensitivityY;
                rotationX = ClampAngle(rotationX, minimumX, maximumX);
                rotationY = ClampAngle(rotationY, minimumY, maximumY);
                Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
                Quaternion yQuaternion = Quaternion.AngleAxis(rotationY, -Vector3.right);

                transform.localRotation = originalRotation * xQuaternion * yQuaternion;
                // print("look");
        }

        public bool applyStandingForce = false;
        public float standingForce = 10;
        public ForceMode standingForceForceMode;
        public float standingForcePositionOffset = .5f;
        void FixedUpdate()
        {
            if (!allowHumanInput)
            {
                return;
            }
            if (UseMouseRotation)
            {
                if (axes == RotationAxes.MouseXAndY)
                {
                    Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
                    Quaternion yQuaternion = Quaternion.AngleAxis(rotationY, -Vector3.right);

                    transform.localRotation = originalRotation * xQuaternion * yQuaternion;
                }
                else if (axes == RotationAxes.MouseX)
                {
                    // rotationX += Input.GetAxis("Mouse X") * sensitivityX;
                    // rotationX = ClampAngle(rotationX, minimumX, maximumX);

                    Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
                    transform.localRotation = originalRotation * xQuaternion;
                }
                // else
                // {
                //     // rotationY += Input.GetAxis("Mouse Y") * sensitivityY;
                //     // rotationY = ClampAngle(rotationY, minimumY, maximumY);
                //
                //     Quaternion yQuaternion = Quaternion.AngleAxis(-rotationY, Vector3.right);
                //     transform.localRotation = originalRotation * yQuaternion;
                // }

            }

            if (groundCheck && !groundCheck.isGrounded)
            {
                AddFallingForce(rb);
                //                print("AddFallingForce");

            }
            strafeCoolDownTimer += Time.fixedDeltaTime;
            dashCoolDownTimer += Time.fixedDeltaTime;

            if (applyStandingForce)
            {
                //STANDING FORCES
                rb.AddForceAtPosition(Vector3.up * standingForce, transform.TransformPoint(Vector3.up * standingForcePositionOffset),
                    standingForceForceMode);
                rb.AddForceAtPosition(-Vector3.up * standingForce, transform.TransformPoint(-Vector3.up * standingForcePositionOffset),
                    standingForceForceMode);

            }

            if (spinAttack)
            {
                //                rb.AddTorque(Vector3.up * spinAttackSpeed);
                rb.angularVelocity = Vector3.up * spinAttackSpeed;
            }

            var moveDir = transform.TransformDirection(new Vector3(inputH, 0, inputV));


                if (groundCheck.isGrounded)
                {
                    RunOnGround(moveDir);
                }
                // else
                // {
                //     RunInAir(dir.normalized);
                // }

            if (inputH == 0 && inputV == 0)
            {
                //                if (groundCheck && groundCheck.isGrounded && !dashPressed)
                if (groundCheck && groundCheck.isGrounded)
                {
                    AddIdleDrag(rb);
                    //                    print("AddIdleDrag");

                }

            }
        }

        public void Jump()
        {
            Vector3 velToUse = rb.velocity;
            velToUse.y = agentJumpVelocity;
            rb.velocity = velToUse;
        }



        public void RotateBody(Rigidbody rb, Vector3 rotationAxis)
        {
            rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
        }

        public void Strafe(Vector3 dir)
        {
            if (dir != Vector3.zero && strafeCoolDownTimer > strafeCoolDownDuration)
            {
                rb.velocity = Vector3.zero;
                rb.AddForce(dir.normalized * strafeSpeed, strafeForceMode);
                strafeCoolDownTimer = 0;
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
                // print($"rot {gameObject.name} {dir.normalized}");
            }
        }

        public void RunOnGround(Vector3 dir)
        {
                // print(dir);
            if (dir == Vector3.zero)
            {
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero;
                }
            }
            else
            {
                var vel = rb.velocity.magnitude;
                float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentTerminalVel);
                //                float adjustedSpeed = Mathf.MoveTowards(vel, agentRunSpeed, WalkSmoothing);
                //                float adjustedSpeed = Mathf.SmoothDamp(vel, agentRunSpeed, ref agentVel, WalkSmoothing, agentTerminalVel);

                //                rb.AddForce(dir.normalized * adjustedSpeed, runningForceMode);
                rb.AddForce(dir * adjustedSpeed, runningForceMode);
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero +
                                             Vector3.up * walkingAnimScale * walkingBounceCurve.Evaluate(
                                                 m_animateBodyMeshCurveTimer);
                    m_animateBodyMeshCurveTimer += Time.fixedDeltaTime;
                }
                //            rb.AddForceAtPosition(dir.normalized * adjustedSpeed,transform.TransformPoint(Vector3.forward * standingForcePositionOffset),
                //                runningForceMode);

            }
        }
        public void RunOnGround(Rigidbody rb, Vector3 dir)
        {
            print(dir);

            if (dir == Vector3.zero)
            {
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero;
                }
            }
            else
            {
                var vel = rb.velocity.magnitude;
                float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentTerminalVel);
                //                rb.AddForce(dir.normalized * adjustedSpeed, runningForceMode);
                rb.AddForce(dir * adjustedSpeed, runningForceMode);
                if (AnimateBodyMesh)
                {
                    bodyMesh.localPosition = Vector3.zero +
                                             Vector3.up * walkingAnimScale * walkingBounceCurve.Evaluate(
                                                 m_animateBodyMeshCurveTimer);
                    m_animateBodyMeshCurveTimer += Time.fixedDeltaTime;
                }
                //            rb.AddForceAtPosition(dir.normalized * adjustedSpeed,transform.TransformPoint(Vector3.forward * standingForcePositionOffset),
                //                runningForceMode);

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
