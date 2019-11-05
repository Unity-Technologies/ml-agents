using System;
using UnityEngine;

namespace MLAgents
{
    public class ArticulatedCrawlerManualControl : MonoBehaviour
    {
        public GameObject upperLeg;
        public GameObject foreLeg;
        private ArticulationBody m_AbUpper;
        private ArticulationBody m_AbFore;

        private Vector3 m_rotationUpper;
        private Vector3 m_rotationFore;

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
            m_AbUpper = upperLeg.GetComponent<ArticulationBody>();
            m_AbFore = foreLeg.GetComponent<ArticulationBody>();


            m_rotationUpper = m_rotationFore = Vector3.zero;
        }

        /// <summary>
        /// Resets the position and velocity of the agent and the goal.
        /// </summary>
        public void AgentReset()
        {
            m_rotationUpper = Vector3.zero;
            m_rotationFore = Vector3.zero;

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
            float deltaRotation = 0.01f;
            
            
            // upper arm
            if (Input.GetKey(KeyCode.A))
                m_rotationUpper.x += deltaRotation;
            if (Input.GetKey(KeyCode.Z))
                m_rotationUpper.x -= deltaRotation;
            if (Input.GetKey(KeyCode.S))
                m_rotationUpper.y += deltaRotation;
            if (Input.GetKey(KeyCode.X))
                m_rotationUpper.y -= deltaRotation;
            if (Input.GetKey(KeyCode.D))
                m_rotationUpper.z += deltaRotation;
            if (Input.GetKey(KeyCode.C))
                m_rotationUpper.z -= deltaRotation;

            
            m_rotationUpper.x = Mathf.Clamp(m_rotationUpper.x, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);
            m_rotationUpper.y = Mathf.Clamp(m_rotationUpper.y, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);
            m_rotationUpper.z = Mathf.Clamp(m_rotationUpper.z, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);

            
            // lower arm
            if (Input.GetKey(KeyCode.F))
                m_rotationFore.x += deltaRotation;
            if (Input.GetKey(KeyCode.V))
                m_rotationFore.x -= deltaRotation;
            if (Input.GetKey(KeyCode.G))
                m_rotationFore.y += deltaRotation;
            if (Input.GetKey(KeyCode.B))
                m_rotationFore.y -= deltaRotation;
            if (Input.GetKey(KeyCode.H))
                m_rotationFore.z += deltaRotation;
            if (Input.GetKey(KeyCode.N))
                m_rotationFore.z -= deltaRotation;

            m_rotationFore.x = Mathf.Clamp(m_rotationFore.x, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);
            m_rotationFore.y = Mathf.Clamp(m_rotationFore.y, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);
            m_rotationFore.z = Mathf.Clamp(m_rotationFore.z, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);

            
            SetJointTargetRotation(m_AbUpper, m_rotationUpper.x, m_rotationUpper.y, m_rotationUpper.z);
            SetJointStrength(m_AbUpper, 0.0001f);
            
            SetJointTargetRotation(m_AbFore, m_rotationFore.x, m_rotationFore.y, m_rotationFore.z);
            SetJointStrength(m_AbFore, 0.0001f);


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