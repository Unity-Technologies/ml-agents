//Standardized movement controller for the Agent Cube

using System;
using Unity.Mathematics;
using UnityEngine;

namespace MLAgents
{
    public class AgentCubeMovement : MonoBehaviour
    {

        [Header("INPUT")] public bool allowKeyboardInput = true;
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

        [Header("BODY ROTATION")]
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
        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            groundCheck = GetComponent<AgentCubeGroundCheck>();
            rb.maxAngularVelocity = maxAngularVel;
        }

        void Update()
        {
            if (!allowKeyboardInput)
            {
                return;
            }
            var camForward = cam.transform.forward;
            camForward.y = 0;
            var camRight = cam.transform.right;
            //            lookDir = Vector3.zero;
            //            lookDir += Input.GetKey(KeyCode.W) ? Vector3.forward : Vector3.zero;
            //            lookDir += Input.GetKey(KeyCode.S) ? Vector3.back : Vector3.zero;
            //            lookDir += Input.GetKey(KeyCode.D) ? Vector3.right : Vector3.zero;
            //            lookDir += Input.GetKey(KeyCode.A) ? Vector3.left : Vector3.zero;

            //BODY ROTATION
            lookDir = Input.GetAxisRaw("Horizontal");

            //FORWARD MOVEMENT
            inputV = Input.GetAxisRaw("Vertical");

            //LATERAL MOVEMENT
            inputH = 0;
            //            inputH += Input.GetKey(KeyCode.Q) ? -1 : 0;
            //            inputH += Input.GetKey(KeyCode.E) ? 1 : 0;
            inputH += Input.GetKeyDown(KeyCode.Q) ? -1 : 0;
            inputH += Input.GetKeyDown(KeyCode.E) ? 1 : 0;

            //            lookDir = Input.GetKey(KeyCode.A)?
            //            var moveLateral = Vector3.zero;
            //            //FORWARD MOVEMENT
            //            moveforward += Input.GetKey(KeyCode.W) ? Vector3.forward : Vector3.zero;
            //            moveforward += Input.GetKey(KeyCode.S) ? Vector3.back : Vector3.zero;
            //
            //            //LATERAL MOVEMENT
            //            moveLateral += Input.GetKey(KeyCode.Q) ? Vector3.right : Vector3.zero;
            //            moveLateral += Input.GetKey(KeyCode.E) ? Vector3.left : Vector3.zero;
            //
            //            //BODY ROTATION
            //            lookDir += Input.GetKey(KeyCode.D) ? Vector3.right : Vector3.zero;
            //            lookDir += Input.GetKey(KeyCode.A) ? Vector3.left : Vector3.zero;

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

            spinAttack = Input.GetKey(KeyCode.H);
            dashPressed = Input.GetKeyDown(KeyCode.K);

            if (dashPressed)
            {
                //                dashPressed = false;
                //                rb.AddForce(rb.transform.forward * dashBoostForce, dashForceMode);
                rb.AddTorque(rb.transform.right * dashBoostForce, dashForceMode);
                print("dashPressed");
            }
            //            if (Input.GetKey(KeyCode.D))
            //            {
            //                discreteActionsOut[0] = 3;
            //            }
            //            else if (Input.GetKey(KeyCode.W))
            //            {
            //                discreteActionsOut[0] = 1;
            //            }
            //            else if (Input.GetKey(KeyCode.A))
            //            {
            //                discreteActionsOut[0] = 4;
            //            }
            //            else if (Input.GetKey(KeyCode.S))
            //            {
            //                discreteActionsOut[0] = 2;
            //            }
        }


        //        private float yaw;
        //        private float pitch;
        //        public float mouseSensitivity;
        //        public float mouseSensitivityMultiplier;
        //        public float maxMouseSmoothTime;
        //        public float mouseSmoothing;
        //        public Vector2 pitchMinMax;
        //        public float smoothPitch;
        //        public float pitchSmoothV;
        //        public float smoothYaw;
        //        public float yawSmoothV;
        //        public void RotateBody(float rotateAxis, float forwardAxis)
        //        {
        //            // Look input
        //            yaw += Input.GetAxisRaw ("Mouse X") * mouseSensitivity / 10 * mouseSensitivityMultiplier;
        //            pitch -= Input.GetAxisRaw ("Mouse Y") * mouseSensitivity / 10 * mouseSensitivityMultiplier;
        //            pitch = Mathf.Clamp (pitch, pitchMinMax.x, pitchMinMax.y);
        //            float mouseSmoothTime = Mathf.Lerp (0.01f, maxMouseSmoothTime, mouseSmoothing);
        //            smoothPitch = Mathf.SmoothDampAngle (smoothPitch, pitch, ref pitchSmoothV, mouseSmoothTime);
        //            float smoothYawOld = smoothYaw;
        //            smoothYaw = Mathf.SmoothDampAngle (smoothYaw, yaw, ref yawSmoothV, mouseSmoothTime);
        //            if (!debug_playerFrozen && Time.timeScale > 0) {
        //                cam.transform.localEulerAngles = Vector3.right * smoothPitch;
        //                transform.Rotate (Vector3.up * Mathf.DeltaAngle (smoothYawOld, smoothYaw), Space.Self);
        //            }
        //            rb.MoveRotation(rb.rotation * amount);
        //
        //        }


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

        // private void Look(Vector2 rotate)
        // {
        //     if (rotate.sqrMagnitude < 0.01)
        //         return;
        //     var scaledRotateSpeed = agentRotationSpeed * Time.deltaTime;
        //     m_Rotation.y += rotate.x * scaledRotateSpeed;
        //     m_Rotation.x = Mathf.Clamp(m_Rotation.x - rotate.y * scaledRotateSpeed, -89, 89);
        //     transform.localEulerAngles = m_Rotation;
        // }

        public void Look(float rotate)
        {
            if (Mathf.Abs(rotate) < 0.01)
                return;
            // var scaledRotateSpeed = agentRotationSpeed * Time.deltaTime;
            var scaledRotateSpeed = agentRotationSpeed * Time.fixedDeltaTime;
            m_Rotation.y += rotate * scaledRotateSpeed;
            // m_Rotation.x = Mathf.Clamp(m_Rotation.x - rotate.y * scaledRotateSpeed, -89, 89);
            transform.localEulerAngles = m_Rotation;
        }

        public bool applyStandingForce = false;
        public float standingForce = 10;
        public ForceMode standingForceForceMode;
        public float standingForcePositionOffset = .5f;
        void FixedUpdate()
        {

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
            if (!allowKeyboardInput)
            {
                return;
            }





            if (spinAttack)
            {
                //                rb.AddTorque(Vector3.up * spinAttackSpeed);
                rb.angularVelocity = Vector3.up * spinAttackSpeed;
            }

            //            if (inputH != 0 || inputV != 0)
            //            {
            //
            ////                var dir = cam.transform.TransformDirection(new Vector3(inputH, 0, inputV));
            //                var dir = cam.transform.TransformDirection(new Vector3(0, 0, inputV));
            //                //                dir.y = 0;
            //                //HANDLE WALKING
            //                if (groundCheck.isGrounded)
            //                {
            //                    RunOnGround(rb, dir.normalized);
            //                    //                    print("running");
            //                }
            //
            //            }
            if (lookDir != 0)
            {
                if (!spinAttack)
                {
                    ////                    var rot = rb.rotation * Quaternion.Euler(0, lookDir * agentRotationSpeed * Time.fixedDeltaTime, 0);
                    //                    var walkingBackwardsCoeff = 1;
                    //                    if (invertRotationIfWalkingBackwards && inputV < 0)
                    //                    {
                    //                        walkingBackwardsCoeff = -1;
                    //                    }
                    //
                    ////                    var rot = rb.rotation * Quaternion.Euler(0, lookDir * agentRotationSpeed * walkingBackwardsCoeff * Time.fixedDeltaTime, 0);
                    ////                    rb.MoveRotation(rot);
                    ////                    var rot = rb.rotation * Quaternion.Euler(0, lookDir * agentRotationSpeed * walkingBackwardsCoeff * Time.fixedDeltaTime, 0);
                    //                    rb.MoveRotation( Quaternion.Euler(0, lookDir * agentRotationSpeed * walkingBackwardsCoeff * Time.fixedDeltaTime, 0));

                    RotateBody(lookDir, inputV);
                    //                    print("rotating");

                    //                    rb.rotation *= Quaternion.AngleAxis(); rb.rotation * Quaternion. (rb.rotation, rot, agentRotationSpeed * Time.fixedDeltaTime);
                }

            }
            if (inputH != 0)
            {

                Strafe(transform.right * inputH);
            }

            if (inputV != 0)
            {

                //                var dir = cam.transform.TransformDirection(new Vector3(inputH, 0, inputV));
                //                var dir = cam.transform.TransformDirection(new Vector3(0, 0, inputV));
                var dir = transform.forward * inputV;
                //                dir.y = 0;
                //HANDLE WALKING
                if (groundCheck.isGrounded)
                {
                    RunOnGround(rb, dir.normalized);
                    //                    if (AnimateBodyMesh)
                    //                    {
                    //                        bodyMesh.localPosition = Vector3.zero +
                    //                                                 Vector3.up * walkingAnimScale * walkingBounceCurve.Evaluate(
                    //                                                     m_animateBodyMeshCurveTimer);
                    //                        m_animateBodyMeshCurveTimer += Time.fixedDeltaTime;
                    //                    }
                    //                    print("running");
                }
                else
                {
                    RunInAir(rb, dir.normalized);
                }
            }
            //            else //is idle
            //            {
            //
            //
            //            }

            if (inputH == 0 && inputV == 0)
            {
                //                if (groundCheck && groundCheck.isGrounded && !dashPressed)
                if (groundCheck && groundCheck.isGrounded)
                {
                    AddIdleDrag(rb);
                    //                    print("AddIdleDrag");

                }

            }


            //            if (lookDir != Vector3.zero)
            //            {
            //                //                var camRotDir = lookDir;
            //                //                camRotDir.z = Mathf.Clamp01(camRotDir.z);
            //                //                var rot = quaternion.LookRotation(camRotDir, Vector3.up);
            //
            //
            //                var dir = cam.transform.TransformDirection(lookDir);
            //                dir.y = 0;
            //                var rot = quaternion.LookRotation(dir, Vector3.up);
            //                if (!spinAttack)
            //                {
            //                    rb.rotation = Quaternion.Lerp(rb.rotation, rot, agentRotationSpeed * Time.fixedDeltaTime);
            //                }
            //                //                RunOnGround(rb, dir.normalized);
            //                //                var dirToGo = rb.transform.forward;
            //                var dirToGo = dir;
            //                //                RunOnGround(rb, dirToGo);
            //                if (!groundCheck.isGrounded)
            //                {
            //                    //                    RunInAir(rb, dirToGo.normalized);
            //                }
            //                else
            //                {
            //                    //                    var forwardMovement = lookDir;
            //                    ////                    forwardMovement.x = 0; //
            //                    //                    var walkDir = cam.transform.TransformDirection(forwardMovement);
            //                    //                    RunOnGround(rb, walkDir.normalized);
            //
            //                    RunOnGround(rb, dirToGo.normalized);
            //                }
            //                //                rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
            //            }
            //            else //is idle
            //            {
            //                if (groundCheck && groundCheck.isGrounded && !dashPressed)
            //                {
            //                    AddIdleDrag(rb);
            //                }
            //            }




        }
        //        void FixedUpdate()
        //        {
        //            if (spinAttack)
        //            {
        //                //                rb.AddTorque(Vector3.up * spinAttackSpeed);
        //                rb.angularVelocity = Vector3.up * spinAttackSpeed;
        //            }
        //
        //            //HANDLE WALKING
        //            if (groundCheck.isGrounded)
        //            {
        //                RunOnGround(rb, dirToGo.normalized);
        //            }
        //
        //
        //            if (lookDir != Vector3.zero)
        //            {
        //                //                var camRotDir = lookDir;
        //                //                camRotDir.z = Mathf.Clamp01(camRotDir.z);
        //                //                var rot = quaternion.LookRotation(camRotDir, Vector3.up);
        //
        //
        //                var dir = cam.transform.TransformDirection(lookDir);
        //                dir.y = 0;
        //                var rot = quaternion.LookRotation(dir, Vector3.up);
        //                if (!spinAttack)
        //                {
        //                    rb.rotation = Quaternion.Lerp(rb.rotation, rot, agentRotationSpeed * Time.fixedDeltaTime);
        //                }
        //                //                RunOnGround(rb, dir.normalized);
        //                //                var dirToGo = rb.transform.forward;
        //                var dirToGo = dir;
        //                //                RunOnGround(rb, dirToGo);
        //                if (!groundCheck.isGrounded)
        //                {
        //                    //                    RunInAir(rb, dirToGo.normalized);
        //                }
        //                else
        //                {
        //                    //                    var forwardMovement = lookDir;
        //                    ////                    forwardMovement.x = 0; //
        //                    //                    var walkDir = cam.transform.TransformDirection(forwardMovement);
        //                    //                    RunOnGround(rb, walkDir.normalized);
        //
        //                    RunOnGround(rb, dirToGo.normalized);
        //                }
        //                //                rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
        //            }
        //            else //is idle
        //            {
        //                if (groundCheck && groundCheck.isGrounded && !dashPressed)
        //                {
        //                    AddIdleDrag(rb);
        //                }
        //            }
        //
        //            if (groundCheck && !groundCheck.isGrounded)
        //            {
        //                AddFallingForce(rb);
        //            }
        //
        //
        //        }

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
            }
        }

        //        public float WalkSmoothing = 3;
        //        private float agentVel;
        public void RunOnGround(Vector3 dir)
        {
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
        //        public void RunOnGround(Rigidbody rb, Vector3 dir)
        //        {
        //            if (dir == Vector3.zero)
        //            {
        //                if (AnimateBodyMesh)
        //                {
        //                    bodyMesh.localPosition = Vector3.zero;
        //                }
        //            }
        //            else
        //            {
        //                var vel = rb.velocity.magnitude;
        //                float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentTerminalVel);
        //                rb.AddForce(dir.normalized * adjustedSpeed,
        //                    runningForceMode);
        //                if (AnimateBodyMesh)
        //                {
        //                    bodyMesh.localPosition = Vector3.zero +
        //                                             Vector3.up * walkingAnimScale * walkingBounceCurve.Evaluate(
        //                                                 m_animateBodyMeshCurveTimer);
        //                    m_animateBodyMeshCurveTimer += Time.fixedDeltaTime;
        //                }
        //                //            rb.AddForceAtPosition(dir.normalized * adjustedSpeed,transform.TransformPoint(Vector3.forward * standingForcePositionOffset),
        //                //                runningForceMode);
        //
        //            }
        //        }

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
