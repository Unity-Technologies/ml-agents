using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAgent : Agent {

    [Header("Target To Walk Towards")] 
    [Space(10)] 
    public Transform target;
    public bool respawnTargetWhenTouched;
    public float targetSpawnRadius;


    [Header("Body Parts")] 
    [Space(10)] 
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
    public List<BodyPart> bodyPartsList = new List<BodyPart>();



    [Header("Joint Settings")] 
    [Space(10)] 
	public float maxJointSpring;
	public float jointDampen;
	public float maxJointForceLimit;
	public Vector3 footCenterOfMassShift; //used to shift the centerOfMass on the feet so the agent isn't so top heavy
	Vector3 dirToTarget;
	float movingTowardsDot;
	float facingDot;


    [Header("Reward Functions To Use")] 
    [Space(10)] 
    public bool rewardMovingTowardsTarget; //agent should move towards target
    public bool rewardFacingTarget; //agent should face the target
    public bool rewardUseTimePenalty; //hurry up


    [Header("Foot Grounded Visualization")] 
    [Space(10)] 
    public bool useFootGroundedVisualization;
    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;


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
        public CrawlerContact groundContact;
		public CrawlerAgent agent;
        [HideInInspector]
        public Vector3 previousJointRotation;
        [HideInInspector]
        public float previousSpringValue;
        public Vector3 currentJointForce;
        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;

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
                positionSpring = ((strength + 1f) * 0.5f) * agent.maxJointSpring,
				positionDamper = agent.jointDampen,
                maximumForce = agent.maxJointForceLimit
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
        bp.groundContact = t.GetComponent<CrawlerContact>();
		bp.agent = this;
        bodyPartsList.Add(bp);

    }

    //Initialize
    public override void InitializeAgent()
    {
        //Setup each body part
        SetupBodyPart(body);
        SetupBodyPart(leg0_upper);
        SetupBodyPart(leg0_lower);
        SetupBodyPart(leg1_upper);
        SetupBodyPart(leg1_lower);
        SetupBodyPart(leg2_upper);
        SetupBodyPart(leg2_lower);
        SetupBodyPart(leg3_upper);
        SetupBodyPart(leg3_lower);

        //we want a lower center of mass or the crawler will roll over easily. 
        //these settings shift the COM on the lower legs
		bodyParts[leg0_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg1_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg2_lower].rb.centerOfMass = footCenterOfMassShift;
		bodyParts[leg3_lower].rb.centerOfMass = footCenterOfMassShift;
    }


    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp)
    {
        var rb = bp.rb;
        AddVectorObs(bp.groundContact.touchingGround ? 1 : 0); // Is this bp touching the ground

        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToBody = body.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToBody);

        if(bp.rb.transform != body)
        {
            AddVectorObs(Quaternion.FromToRotation(body.forward, bp.rb.transform.forward));
        }
    }

    // void FixedUpdate()
    // {
    //     //update pos to target
	// 	dirToTarget = target.position - bodyParts[body].rb.position;

    //     //if enabled the feet will light up green when the foot is grounded.
    //     //this is just a visualization and isn't necessary for function
    //     if(useFootGroundedVisualization)
    //     {
    //         foot0.material = bodyParts[leg0_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
    //         foot1.material = bodyParts[leg1_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
    //         foot2.material = bodyParts[leg2_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
    //         foot3.material = bodyParts[leg3_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
    //     }
    // }

	/// <summary>
    /// Adds the raycast hit dist and relative pos to observations
    /// </summary>
    void RaycastObservation(Vector3 pos, Vector3 dir, float maxDist)
    {
        RaycastHit hit;
        float dist = 0;
        Vector3 relativeHitPos = Vector3.zero;
        if(Physics.Raycast(pos, dir, out hit, maxDist))
        {
            if(hit.collider.CompareTag("ground"))
            {
                //normalized hit distance
                dist = hit.distance/maxDist; 

                //hit point position relative to the body's local space
                relativeHitPos = body.InverseTransformPoint(hit.point); 
            }
        }

        //add our raycast observation 
        AddVectorObs(dist);
        AddVectorObs(relativeHitPos);
    }

    public override void CollectObservations()
    {
        //normalize dir vector to help generalize
        AddVectorObs(dirToTarget.normalized);

        //raycast out of the bottom of the legs to get information about where the ground is
        RaycastObservation(leg0_lower.position, leg0_lower.up, 5);
        RaycastObservation(leg1_lower.position, leg1_lower.up, 5);
        RaycastObservation(leg2_lower.position, leg2_lower.up, 5);
        RaycastObservation(leg3_lower.position, leg3_lower.up, 5);

        //forward & up to help with orientation
        AddVectorObs(body.forward);
        AddVectorObs(body.up);
        GetCurrentJointForces();
        foreach (var bodyPart in bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }

    void GetCurrentJointForces()
    {
        foreach (var bodyPart in bodyParts.Values)
        {
            if(bodyPart.joint)
            {
                bodyPart.currentJointForce = bodyPart.joint.currentForce;
                bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.sqrMagnitude;
                bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
                bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.sqrMagnitude;
            }
        }
    }

	/// <summary>
    /// Agent touched the target
    /// </summary>
	public void TouchedTarget(float impactForce)
	{
		AddReward(.01f * impactForce); //higher impact should be rewarded
        if(respawnTargetWhenTouched)
        {
		    GetRandomTargetPos();
        }
		Done();
	}

    /// <summary>
    /// Moves target to a random position within specified radius.
    /// </summary>
    /// <returns>
    /// Move target to random position.
    /// </returns>
    public void GetRandomTargetPos()
    {
        Vector3 newTargetPos = Random.insideUnitSphere * targetSpawnRadius;
		newTargetPos.y = 5;
		target.position = newTargetPos;
    }

	 public override void AgentAction(float[] vectorAction, string textAction)
    {
        //update pos to target
		dirToTarget = target.position - bodyParts[body].rb.position;

        //if enabled the feet will light up green when the foot is grounded.
        //this is just a visualization and isn't necessary for function
        if(useFootGroundedVisualization)
        {
            foot0.material = bodyParts[leg0_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot1.material = bodyParts[leg1_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot2.material = bodyParts[leg2_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot3.material = bodyParts[leg3_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
        }

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
        if(rewardMovingTowardsTarget){RewardFunctionMovingTowards();}
        if(rewardFacingTarget){RewardFunctionFacingTarget();}
        if(rewardUseTimePenalty){RewardFunctionTimePenalty();}
    }
	
    //Reward moving towards target & Penalize moving away from target.
    void RewardFunctionMovingTowards()
    {
        //don't normalize vel. the faster it goes the more reward it should get
        //0.03f chosen via experimentation
		movingTowardsDot = Vector3.Dot(bodyParts[body].rb.velocity, dirToTarget.normalized); 
        AddReward(0.03f * movingTowardsDot);
    }

    //Reward facing target & Penalize facing away from target
    void RewardFunctionFacingTarget()
    {
        //0.01f chosen via experimentation.
		facingDot = Vector3.Dot(dirToTarget.normalized, body.forward);
        AddReward(0.01f * facingDot);
    }

    //Time penalty - HURRY UP
    void RewardFunctionTimePenalty()
    {
        //0.001f chosen by experimentation. If this penalty is too high it will kill itself :(
        AddReward(- 0.001f); 
    }

	/// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void AgentReset()
    {
        if(dirToTarget != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(dirToTarget);
        }
        
        foreach (var bodyPart in bodyParts.Values)
        {
            bodyPart.Reset();
        }
    }
}