using System.Collections;
using System.Collections.Generic;
using UnityEngine;




    /// <summary>
    /// Used to store relevant information for acting and learning for each body part in agent.
    /// </summary>
    [System.Serializable]
    public class BodyPart
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public Vector3 startingPos;
        public Quaternion startingRot;
        public GroundContact groundContact;
        public TargetContact targetContact;
		// public CrawlerAgent agent;
        [HideInInspector]
        public Vector3 previousJointRotation;
        [HideInInspector]
        public float previousSpringValue;
        public Vector3 currentJointForce;
        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;
    }

public class JointDriveController : MonoBehaviour {


    [Header("Joint Settings")] 
    [Space(10)] 
	public float maxJointSpring;
	public float jointDampen;
	public float maxJointForceLimit;
    // [Tooltip("Reward Functions To Use")] 

    public float maxJointAngleChangePerDecision; //the change in joint angle will not be able to exceed this value.
    public float maxJointStrengthChangePerDecision; //the change in joint strenth will not be able to exceed this value.
	// public Vector3 footCenterOfMassShift; //used to shift the centerOfMass on the feet so the agent isn't so top heavy
	// Vector3 dirToTarget;
	// float movingTowardsDot;
	float facingDot;




	// Use this for initialization
	void Start () {
		
	}
        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset(BodyPart bp)
        {
            print("resetting bp");
            bp.rb.transform.position = bp.startingPos;
            bp.rb.transform.rotation = bp.startingRot;
            bp.rb.velocity = Vector3.zero;
            bp.rb.angularVelocity = Vector3.zero;
			bp.groundContact.touchingGround = false;;
        }
	
            /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetNormalizedTargetRotation(BodyPart bp, float x, float y, float z)
        {
            // var bp = bodyParts[t];
            // Transform values from [-1, 1] to [0, 1]
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xRot = Mathf.MoveTowards(bp.previousJointRotation.x, Mathf.Lerp(bp.joint.lowAngularXLimit.limit, bp.joint.highAngularXLimit.limit, x),maxJointAngleChangePerDecision);
            var yRot = Mathf.MoveTowards(bp.previousJointRotation.y, Mathf.Lerp(-bp.joint.angularYLimit.limit, bp.joint.angularYLimit.limit, y), maxJointAngleChangePerDecision);
            var zRot = Mathf.MoveTowards(bp.previousJointRotation.z, Mathf.Lerp(-bp.joint.angularZLimit.limit, bp.joint.angularZLimit.limit, z), maxJointAngleChangePerDecision);

            // var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
            // var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
            // var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

            bp.joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
            bp.previousJointRotation = new Vector3(xRot, yRot, zRot);

            // var jd = new JointDrive
            // {
            //     positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
			// 	positionDamper = agent.jointDampen,
            //     maximumForce = agent.maxJointForceLimit
            // };
            // joint.slerpDrive = jd;
        }



        public void UpdateJointDrive(BodyPart bp, float strength)
        {
            // var bp = bodyParts[t];
            // var spring = Mathf.MoveTowards(previousSpringValue, (strength + 1f) * 0.5f, agent.maxJointStrengthChangePerDecision);
            var rawSpringVal = ((strength + 1f) * 0.5f) * maxJointSpring;
            var clampedSpring = Mathf.MoveTowards(bp.previousSpringValue, rawSpringVal, maxJointStrengthChangePerDecision);
            // agent.energyPenalty += clampedSpring/agent.maxJointStrengthChangePerDecision;
            var jd = new JointDrive
            {
                // positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
                // positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
                // positionSpring = spring * agent.maxJointSpring,
                positionSpring = clampedSpring,
                positionDamper = jointDampen,
                maximumForce = maxJointForceLimit
            };
            bp.joint.slerpDrive = jd;

            // previousJointRotation = new Vector3(xRot, yRot, zRot);
            bp.previousSpringValue = jd.positionSpring;
        }


	// Update is called once per frame
	void Update () {
		
	}
}
