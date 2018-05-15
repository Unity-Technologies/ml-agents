using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAgent : Agent {
    [Header("Specific to Walker")] 
    public Vector3 goalDirection;
    public Transform body;
    public Transform leg0_upper;
    public Transform leg0_lower;
    public Transform leg1_upper;
    public Transform leg1_lower;
    public Transform leg2_upper;
    public Transform leg2_lower;
    public Transform leg3_upper;
    public Transform leg3_lower;
    public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();
	public Vector3 dirToTarget;
	CrawlerAcademy academy;
	public float movingTowardsDot;
	public float facingDot;
	// public float prevDistSqrMag;
	public float maxJointSpring;
	public float jointDampen;
	public float maxJointForceLimit;
	public Vector3 footCenterOfMassShift;
	// public bool useMoveTowardTargetRewardFunction;

	// enum RewardFunctions
	// {
	// 	walkTowardsDir
	// }

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
		public CrawlerAgent agent;

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset()
        {
            rb.transform.position = startingPos;
            rb.transform.rotation = startingRot;
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
			groundContact.touchingGround = false;;
        }
        
        /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetNormalizedTargetRotation(float x, float y, float z, float strength)
        {
            // Transform values from [-1, 1] to [0, 1]
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

            joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
            var jd = new JointDrive
            {
                // positionSpring = ((strength + 1f) * 0.5f) * 10000f,
                positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
				// positionDamper = agent.maxJointSpring * .075f, //chosen via experimentation
				positionDamper = agent.jointDampen,
                maximumForce = agent.maxJointForceLimit
                // maximumForce = 250000f
            };
            joint.slerpDrive = jd;
        }

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
        bodyParts.Add(t, bp);
        bp.groundContact = t.GetComponent<GroundContact>();
		bp.agent = this;
    }
    public override void InitializeAgent()
    {
		academy = FindObjectOfType<CrawlerAcademy>();
        SetupBodyPart(body);
        SetupBodyPart(leg0_upper);
        SetupBodyPart(leg0_lower);
        SetupBodyPart(leg1_upper);
        SetupBodyPart(leg1_lower);
        SetupBodyPart(leg2_upper);
        SetupBodyPart(leg2_lower);
        SetupBodyPart(leg3_upper);
        SetupBodyPart(leg3_lower);
		bodyParts[leg0_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg1_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg2_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg3_lower].rb.centerOfMass = footCenterOfMassShift;
    }

    /// <summary>
    /// Obtains joint rotation (in Quaternion) from joint. 
    /// </summary>
    public static Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return (Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

	
    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp)
    {
        var rb = bp.rb;
        AddVectorObs(bp.groundContact.touchingGround ? 1 : 0); // Is this bp touching the ground
        // bp.groundContact.touchingGround = false;

        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToBody = body.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToBody);

        // if (bp.joint && (bp.rb.transform != handL && bp.rb.transform != handR))
        if (bp.joint)
        {
            var jointRotation = GetJointRotation(bp.joint);
            AddVectorObs(jointRotation); // Get the joint rotation
        }
    }

	/// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations()
    {
		dirToTarget = academy.target.position - bodyParts[body].rb.position;
		// dirToTarget.Normalize();
        AddVectorObs(dirToTarget);
        AddVectorObs(bodyParts[body].rb.rotation);
		// print(bodyParts[body].rb.rotation);
        AddVectorObs(Vector3.Dot(dirToTarget.normalized, body.forward)); //are we facing the target?
        AddVectorObs(Vector3.Dot(bodyParts[body].rb.velocity.normalized, dirToTarget.normalized)); //are we moving towards or away from target?
        // AddVectorObs(body.position.y);
        // AddVectorObs(body.forward);
        // AddVectorObs(body.up);

        // AddVectorObs(goalDirection);
        // AddVectorObs(body.forward);
        // AddVectorObs(body.up);
		
        foreach (var bodyPart in bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }

	public void TouchedTarget(float impactForce)
	{
		AddReward(.01f * impactForce);
		// AddReward(1);
		academy.GetRandomTargetPos();

		Done();
	}

	 public override void AgentAction(float[] vectorAction, string textAction)
    {

		// Debug.DrawRay(bodyParts[leg0_lower].rb.worldCenterOfMass, Vector3.up, Color.red);
        // Apply action to all relevant body parts. 
        
        bodyParts[leg0_upper].SetNormalizedTargetRotation(vectorAction[0], vectorAction[1], 0, vectorAction[2]);
        bodyParts[leg1_upper].SetNormalizedTargetRotation(vectorAction[3], vectorAction[4], 0, vectorAction[5]);
        bodyParts[leg2_upper].SetNormalizedTargetRotation(vectorAction[6], vectorAction[7], 0, vectorAction[8]);
        bodyParts[leg3_upper].SetNormalizedTargetRotation(vectorAction[9], vectorAction[10], 0, vectorAction[11]);
        bodyParts[leg0_lower].SetNormalizedTargetRotation(vectorAction[12], 0, 0, vectorAction[13]);
        bodyParts[leg1_lower].SetNormalizedTargetRotation(vectorAction[14], 0, 0, vectorAction[15]);
        bodyParts[leg2_lower].SetNormalizedTargetRotation(vectorAction[16], 0, 0, vectorAction[17]);
        bodyParts[leg3_lower].SetNormalizedTargetRotation(vectorAction[18], 0, 0, vectorAction[19]);

        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        // d. Discourage head movement.
		// movingTowardsDot = Vector3.Dot(dirToTarget, bodyParts[body].rb.velocity);
		// movingTowardsDot = Vector3.Dot(dirToTarget, bodyParts[body].rb.velocity);
		// movingTowardsDot = Vector3.Dot(dirToTarget.normalized, bodyParts[body].rb.velocity.normalized);


		movingTowardsDot = Vector3.Dot(bodyParts[body].rb.velocity, dirToTarget.normalized); //don't normalize vel. the faster it goes the more reward it should get
		// facingDot = Vector3.Dot(dirToTarget.normalized, body.forward);
		// float currentSqrMag = dirToTarget.sqrMagnitude;
		// float closerReward = 0;
		// if(currentSqrMag < prevDistSqrMag) //getting closer
		// {
		// 	closerReward = .03f;
		// 	// AddReward(.03f);
		// 	// print("closer");
		// }
		// else
		// {
		// 	closerReward = -.03f;
		// 	// AddReward(-.03f);
		// }

        AddReward(
            + 0.03f * movingTowardsDot
			- 0.001f //hurry up
            // + 0.01f * facingDot
			// + closerReward
            // + 0.03f * Vector3.Dot(goalDirection, bodyParts[body].rb.velocity)
            // + 0.01f * Vector3.Dot(goalDirection, body.forward)
            // + 0.01f * (head.position.y - hips.position.y)
            // - 0.01f * Vector3.Distance(bodyParts[head].rb.velocity, bodyParts[hips].rb.velocity)
        );
		// prevDistSqrMag = currentSqrMag;
		// print(this.gameObject.name + " reward: " + GetReward());
    }
	

	/// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void AgentReset()
    {
        transform.rotation = Quaternion.LookRotation(goalDirection);
        
        foreach (var bodyPart in bodyParts.Values)
        {
            bodyPart.Reset();
        }
    }
}
