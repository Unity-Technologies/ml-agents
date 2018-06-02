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
        [HideInInspector]
        public JointDriveController thisJDController;
		// public CrawlerAgent agent;
        [HideInInspector]
        public Vector3 previousJointRotation;
        [HideInInspector]
        public float previousStrengthValue;
        public Vector3 currentJointForce;
        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;
        public AnimationCurve jointForceCurve = new AnimationCurve();
        public AnimationCurve jointTorqueCurve = new AnimationCurve();

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset(BodyPart bp)
        {
            bp.rb.transform.position = bp.startingPos;
            bp.rb.transform.rotation = bp.startingRot;
            bp.rb.velocity = Vector3.zero;
            bp.rb.angularVelocity = Vector3.zero;
            if(bp.groundContact)
            {
			    bp.groundContact.touchingGround = false;
            }
            if(bp.targetContact)
            {
			    bp.targetContact.touchingTarget = false;
            }
        }

         /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetJointTargetRotation(float x, float y, float z)
        {
            // var bp = bodyParts[t];
            // Transform values from [-1, 1] to [0, 1]
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xRot = Mathf.MoveTowards(previousJointRotation.x, Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x), thisJDController.maxJointAngleChangePerDecision);
            var yRot = Mathf.MoveTowards(previousJointRotation.y, Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y), thisJDController.maxJointAngleChangePerDecision);
            var zRot = Mathf.MoveTowards(previousJointRotation.z, Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z), thisJDController.maxJointAngleChangePerDecision);

            // var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
            // var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
            // var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

            joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
            previousJointRotation = new Vector3(xRot, yRot, zRot);

            // var jd = new JointDrive
            // {
            //     positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
			// 	positionDamper = agent.jointDampen,
            //     maximumForce = agent.maxJointForceLimit
            // };
            // joint.slerpDrive = jd;
        }



        public void SetJointStrength(float strength)
        {
            // var bp = bodyParts[t];
            // var spring = Mathf.MoveTowards(previousSpringValue, (strength + 1f) * 0.5f, agent.maxJointStrengthChangePerDecision);
            var rawVal = ((strength + 1f) * 0.5f) * thisJDController.maxJointForceLimit;
            var clampedStrength = Mathf.MoveTowards(previousStrengthValue, rawVal, thisJDController.maxJointStrengthChangePerDecision);
            // var rawSpringVal = ((strength + 1f) * 0.5f) * thisJDController.maxJointSpring;
            // var clampedSpring = Mathf.MoveTowards(previousSpringValue, rawSpringVal, thisJDController.maxJointStrengthChangePerDecision);
            // agent.energyPenalty += clampedSpring/agent.maxJointStrengthChangePerDecision;
            var jd = new JointDrive
            {
                // positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
                // positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
                // positionSpring = spring * agent.maxJointSpring,
                // positionSpring = clampedSpring,
                positionSpring = thisJDController.maxJointSpring,
                positionDamper = thisJDController.jointDampen,
                maximumForce = clampedStrength
                // maximumForce = thisJDController.maxJointForceLimit
            };
            // jd.mode = JointDriveMode.Position;
            joint.slerpDrive = jd;

            // previousJointRotation = new Vector3(xRot, yRot, zRot);
            previousStrengthValue = jd.maximumForce;
            // previousStrengthValue = jd.positionSpring;
        }

    }

public class JointDriveController : MonoBehaviour {

    //These settings are used when updating the JointDrive settings (the joint's strength)
    [Header("Joint Drive Settings")] 
    [Space(10)] 
	public float maxJointSpring;
	public float jointDampen;
	public float maxJointForceLimit;
    // [Tooltip("Reward Functions To Use")] 


    //These settings are used to clamp the amount a joint can change every decision;
    [Header("Max Joint Movement Per Decision")] 
    [Space(10)] 
    public float maxJointAngleChangePerDecision; //the change in joint angle will not be able to exceed this value.
    public float maxJointStrengthChangePerDecision; //the change in joint strenth will not be able to exceed this value.
	// public Vector3 footCenterOfMassShift; //used to shift the centerOfMass on the feet so the agent isn't so top heavy
	// Vector3 dirToTarget;
	// float movingTowardsDot;
	float facingDot;
    public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();
    public List<BodyPart> bodyPartsList = new List<BodyPart>(); //to look at values in inspector, just for debugging




	// Use this for initialization
	void Awake () {
		bodyParts.Clear();
		bodyPartsList.Clear();
	}


    /// <summary>
    /// Create BodyPart object and add it to dictionary.
    /// </summary>
    public void SetupBodyPart(Transform t)
    {
        BodyPart bp = new BodyPart
        {
            rb = t.GetComponent<Rigidbody>(),
            joint = t.GetComponent<ConfigurableJoint>(),
            startingPos = t.position,
            startingRot = t.rotation
        };
		bp.rb.maxAngularVelocity = 100;
        bp.groundContact = t.GetComponent<GroundContact>();
        bp.targetContact = t.GetComponent<TargetContact>();
        bp.thisJDController = this;
		// bp.agent = this;
        bodyParts.Add(t, bp);
        bodyPartsList.Add(bp);
    }

	

    public void GetCurrentJointForces()
    {
        foreach (var bodyPart in bodyParts.Values)
        {
            if(bodyPart.joint)
            {
                bodyPart.currentJointForce = bodyPart.joint.currentForce;
                // bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.sqrMagnitude;
                bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.magnitude;
                bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
                // bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.sqrMagnitude;
                bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.magnitude;
                if (Application.isEditor)
                {
                    if(bodyPart.jointForceCurve.length > 1000)
                    {
                        bodyPart.jointForceCurve = new AnimationCurve();
                    }
                    if(bodyPart.jointTorqueCurve.length > 1000)
                    {
                        bodyPart.jointTorqueCurve = new AnimationCurve();
                    }
                    bodyPart.jointForceCurve.AddKey(Time.time, bodyPart.currentJointForceSqrMag);
                    bodyPart.jointTorqueCurve.AddKey(Time.time, bodyPart.currentJointTorqueSqrMag);
                }
            }
        }
    }

	// Update is called once per frame
	void Update () {
		
	}
}
