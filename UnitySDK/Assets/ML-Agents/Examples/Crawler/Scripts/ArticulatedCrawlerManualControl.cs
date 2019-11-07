using System;
using UnityEngine;

namespace MLAgents
{
    public class ArticulatedCrawlerManualControl : MonoBehaviour
    {
        public GameObject upperLeg0, upperLeg1;
        public GameObject foreLeg0, foreLeg1;
        private ArticulationBody m_AbUpper0, m_AbUpper1;
        private ArticulationBody m_AbFore0, m_AbFore1;

        private Vector3 m_rotationUpper0, m_rotationUpper1;
        private Vector3 m_rotationFore0, m_rotationFore1;

        public bool useAlternativeKeySetForInput = false;
        public float maxJointForceLimit = 1000;
        public float maxJointSpring = 40;
        public float jointDampen = 3000;

        public float currentStrength = 0.0f;

        /// <summary>
        /// Collect the rigidbodies of the reacher in order to resue them for
        /// observations and actions.
        /// </summary>
        public void Start()
        {
            m_AbUpper0 = upperLeg0.GetComponent<ArticulationBody>();
            m_AbFore0 = foreLeg0.GetComponent<ArticulationBody>();

            m_AbUpper1 = upperLeg1.GetComponent<ArticulationBody>();
            m_AbFore1 = foreLeg1.GetComponent<ArticulationBody>();


            m_rotationUpper0 = m_rotationFore0 = m_rotationUpper1 = m_rotationFore1 = Vector3.zero;
        }

        /// <summary>
        /// Resets the position and velocity of the agent and the goal.
        /// </summary>
        public void AgentReset()
        {
            m_rotationUpper0 = m_rotationFore0 = m_rotationUpper1 = m_rotationFore1 = Vector3.zero;

            /*
            m_AbUpper.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
            m_AbFore.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
    
            pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
            pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
            */
        }

        public void FixedUpdate()
        {
            //float maxTorque = 1000f;
            float speed = 100f;
            float deltaRotation = speed * Time.fixedDeltaTime;

            m_rotationUpper0.x = m_AbUpper0.xDrive.target;
            m_rotationUpper0.y = m_AbUpper0.yDrive.target;
            m_rotationUpper0.z = m_AbUpper0.zDrive.target;
            
            m_rotationUpper1.x = m_AbUpper1.xDrive.target;
            m_rotationUpper1.y = m_AbUpper1.yDrive.target;
            m_rotationUpper1.z = m_AbUpper1.zDrive.target;
            
            
            
            // upper arm
            if (Input.GetKey(KeyCode.A))
            {
                m_rotationUpper0.x += deltaRotation;
                m_rotationUpper1.x += deltaRotation;
            }

            if (Input.GetKey(KeyCode.Z))
            {
                m_rotationUpper0.x -= deltaRotation;
                m_rotationUpper1.x -= deltaRotation;
            }

            if (Input.GetKey(KeyCode.S))
            {
                m_rotationUpper0.y += deltaRotation;
                m_rotationUpper1.y += deltaRotation;
            }

            if (Input.GetKey(KeyCode.X))
            {
                m_rotationUpper0.y -= deltaRotation;
                m_rotationUpper1.y -= deltaRotation;
            }

            if (Input.GetKey(KeyCode.D))
            {
                m_rotationUpper0.z += deltaRotation;
                m_rotationUpper1.z += deltaRotation;
            }

            if (Input.GetKey(KeyCode.C))
            {
                m_rotationUpper0.z -= deltaRotation;
                m_rotationUpper1.z -= deltaRotation;
            }


            m_rotationUpper0.x = Mathf.Clamp(m_rotationUpper0.x, -180.0f, 180.0f);
            m_rotationUpper0.y = Mathf.Clamp(m_rotationUpper0.y, -180.0f, 180.0f);
            m_rotationUpper0.z = Mathf.Clamp(m_rotationUpper0.z, -180.0f, 180.0f);

            m_rotationUpper1.x = Mathf.Clamp(m_rotationUpper1.x, -180.0f, 180.0f);
            m_rotationUpper1.y = Mathf.Clamp(m_rotationUpper1.y, -180.0f, 180.0f);
            m_rotationUpper1.z = Mathf.Clamp(m_rotationUpper1.z, -180.0f, 180.0f);

            
            // lower arm
            
            m_rotationFore0.x = m_AbFore0.xDrive.target;
            m_rotationFore0.y = m_AbFore0.yDrive.target;
            m_rotationFore0.z = m_AbFore0.zDrive.target;


            if (Input.GetKey(KeyCode.F))
            {
                m_rotationFore0.x += deltaRotation;
                m_rotationFore1.x += deltaRotation;
            }

            if (Input.GetKey(KeyCode.V))
            {
                m_rotationFore0.x -= deltaRotation;
                m_rotationFore1.x -= deltaRotation;
            }

            if (Input.GetKey(KeyCode.G))
            {
                m_rotationFore0.y += deltaRotation;
                m_rotationFore1.y += deltaRotation;
            }

            if (Input.GetKey(KeyCode.B))
            {
                m_rotationFore0.y -= deltaRotation;
                m_rotationFore1.y -= deltaRotation;
            }

            if (Input.GetKey(KeyCode.H))
            {
                m_rotationFore0.z += deltaRotation;
                m_rotationFore1.z += deltaRotation;
            }

            if (Input.GetKey(KeyCode.N))
            {
                m_rotationFore0.z -= deltaRotation;
                m_rotationFore1.z -= deltaRotation;
            }

            m_rotationFore0.x = Mathf.Clamp(m_rotationFore0.x, -180.0f, 180.0f);
            m_rotationFore0.y = Mathf.Clamp(m_rotationFore0.y, -180.0f, 180.0f);
            m_rotationFore0.z = Mathf.Clamp(m_rotationFore0.z, -180.0f, 180.0f);

            m_rotationFore1.x = Mathf.Clamp(m_rotationFore1.x, -180.0f, 180.0f);
            m_rotationFore1.y = Mathf.Clamp(m_rotationFore1.y, -180.0f, 180.0f);
            m_rotationFore1.z = Mathf.Clamp(m_rotationFore1.z, -180.0f, 180.0f);

            
            SetJointTargetRotation(m_AbUpper0, m_rotationUpper0.x, m_rotationUpper0.y, m_rotationUpper0.z);
            SetJointStrength(m_AbUpper0, 0.0001f);
            
            SetJointTargetRotation(m_AbUpper1, m_rotationUpper1.x, m_rotationUpper1.y, m_rotationUpper1.z);
            SetJointStrength(m_AbUpper1, 0.0001f);

            SetJointTargetRotation(m_AbFore0, m_rotationFore0.x, m_rotationFore0.y, m_rotationFore0.z);
            SetJointStrength(m_AbFore0, 0.0001f);

            SetJointTargetRotation(m_AbFore1, m_rotationFore1.x, m_rotationFore1.y, m_rotationFore1.z);
            SetJointStrength(m_AbFore1, 0.0001f);


            if (Input.GetKey(KeyCode.Escape))
                AgentReset();
        }

        public void SetJointTargetRotation(ArticulationBody arb, float x, float y, float z)
        {
            /*
            // Incoming values need to be in [-1,1] interval
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;
            */
            var xDrive = arb.xDrive;
            var yDrive = arb.yDrive;
            var zDrive = arb.zDrive;
            
            
            /*
            var xRot = Mathf.Lerp(xDrive.lowerLimit, xDrive.upperLimit, x);
            var yRot = Mathf.Lerp(yDrive.lowerLimit, yDrive.upperLimit, y);
            var zRot = Mathf.Lerp(zDrive.lowerLimit, zDrive.upperLimit, z);

            var currentXNormalizedRot =
                Mathf.InverseLerp(xDrive.lowerLimit, xDrive.upperLimit, xRot);
            
            // What is this ? Vilmantas Why lowerLimit is not used ?
            var currentYNormalizedRot = Mathf.InverseLerp(yDrive.lowerLimit, yDrive.upperLimit, yRot);
            var currentZNormalizedRot = Mathf.InverseLerp(zDrive.lowerLimit, zDrive.upperLimit, zRot);

            //joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot); // Original code
            */
            var xRot = x;
            var yRot = y;
            var zRot = z;
            
            xDrive.target = xRot; yDrive.target = yRot; zDrive.target = zRot;

            arb.xDrive = xDrive; arb.yDrive = yDrive; arb.zDrive = zDrive;
            
            var currentEularJointRotation = new Vector3(xRot, yRot, zRot);
        }

        public void SetJointStrength(ArticulationBody arb, float strength)
        {
            var xDrive = arb.xDrive;
            var yDrive = arb.yDrive;
            var zDrive = arb.zDrive;
                
            var rawVal = (strength + 1f) * 0.5f * maxJointForceLimit;
            
            xDrive.stiffness = yDrive.stiffness = zDrive.stiffness = maxJointSpring;
            xDrive.damping = yDrive.damping = zDrive.damping = jointDampen;
            xDrive.forceLimit = yDrive.forceLimit = zDrive.forceLimit = rawVal;

            // Slerp drive does not exist, so we try to set strength for each axis individually
            arb.xDrive = xDrive;
            arb.yDrive = yDrive;
            arb.zDrive = zDrive;
            //joint.slerpDrive = jd;
            currentStrength = rawVal;
        }
    }
}