//Standardized movement controller for the Agent Cube

using System;
using Unity.Mathematics;
using UnityEngine;

namespace MLAgents
{
    public class AgentCubeMovement : MonoBehaviour
    {
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

        [Header("IDLE")]
        //coefficient used to dampen velocity when idle
        //the purpose of this is to fine tune agent drag
        //...and prevent the agent sliding around while grounded
        //0 means it will instantly stop when grounded
        //1 means no drag will be applied
        public float agentIdleDragVelCoeff = .9f;

        [Header("GROUND POUND")]
        public ForceMode groundPoundForceMode = ForceMode.Impulse;
        public float groundPoundForce = 35f;

        [Header("SPIN ATTACK")]
        public float spinAttackSpeed = 20f;
        private bool spinAttack;

        [Header("BODY ROTATION")]
        //body rotation speed
        public float agentRotationSpeed = 35f;

        [Header("JUMPING")]
        //upward jump velocity magnitude
        public float agentJumpVelocity = 15f;

        [Header("FALLING FORCE")]
        //force applied to agent while falling
        public float agentFallingSpeed = 50f;

        public Camera cam;
        private Vector3 lookDir;
        private Rigidbody rb;
        private AgentCubeGroundCheck groundCheck;
        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            groundCheck = GetComponent<AgentCubeGroundCheck>();
            rb.maxAngularVelocity = maxAngularVel;
        }

        void Update()
        {
            var camForward = cam.transform.forward;
            camForward.y = 0;
            var camRight = cam.transform.right;
            lookDir = Vector3.zero;
            lookDir += Input.GetKey(KeyCode.D) ? Vector3.right : Vector3.zero;
            lookDir += Input.GetKey(KeyCode.W) ? Vector3.forward : Vector3.zero;
            lookDir += Input.GetKey(KeyCode.A) ? Vector3.left : Vector3.zero;
            lookDir += Input.GetKey(KeyCode.S) ? Vector3.back : Vector3.zero;


            if (Input.GetKeyDown(KeyCode.Space))
            {
                if (groundCheck)
                {
                    if (groundCheck.isGrounded)
                    {
                        Jump(rb);
                    }
                    else
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

        void FixedUpdate()
        {
            if (spinAttack)
            {
                //                rb.AddTorque(Vector3.up * spinAttackSpeed);
                rb.angularVelocity = Vector3.up * spinAttackSpeed;
            }
            if (lookDir != Vector3.zero)
            {
                var dir = cam.transform.TransformDirection(lookDir);
                dir.y = 0;
                var rot = quaternion.LookRotation(dir, Vector3.up);
                if (!spinAttack)
                {
                    rb.rotation = Quaternion.Lerp(rb.rotation, rot, agentRotationSpeed * Time.deltaTime);
                }
                //                RunOnGround(rb, dir.normalized);
                //                var dirToGo = rb.transform.forward;
                var dirToGo = dir;
                //                RunOnGround(rb, dirToGo);
                if (!groundCheck.isGrounded)
                {
                    //                    RunInAir(rb, dirToGo.normalized);
                }
                else
                {
                    RunOnGround(rb, dirToGo.normalized);
                }
                //                rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
            }
            else //is idle
            {
                if (groundCheck && groundCheck.isGrounded && !dashPressed)
                {
                    AddIdleDrag(rb);
                }
            }

            if (groundCheck && !groundCheck.isGrounded)
            {
                AddFallingForce(rb);
            }


        }

        public void Jump(Rigidbody rb)
        {
            Vector3 velToUse = rb.velocity;
            velToUse.y = agentJumpVelocity;
            rb.velocity = velToUse;
        }

        public void RotateBody(Rigidbody rb, Vector3 rotationAxis)
        {
            rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
        }

        public void RunOnGround(Rigidbody rb, Vector3 dir)
        {
            var vel = rb.velocity.magnitude;
            float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentTerminalVel);
            rb.AddForce(dir.normalized * adjustedSpeed,
                runningForceMode);
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
