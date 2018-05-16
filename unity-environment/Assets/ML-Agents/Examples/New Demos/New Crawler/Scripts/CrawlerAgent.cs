using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAgent : Agent {
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


    [Header("Joint Settings")] 
    [Space(10)] 
	public float maxJointSpring;
	public float jointDampen;
	public float maxJointForceLimit;
	public Vector3 footCenterOfMassShift; //used to shift the centerOfMass on the feet so the agent isn't so top heavy
	Vector3 dirToTarget;
	CrawlerAcademy academy;
	float movingTowardsDot;
	float facingDot;


    [Header("Reward Functions To Use")] 
    [Space(10)] 

    public bool rewardMovingTowardsTarget; //agent should move towards target
    public bool rewardFacingTarget; //agent should face the target
    public bool rewardUseTimePenalty; //hurry up



    [Header("Reward Functions To Use")] 
    [Space(10)] 
    public LayerMask groundLayer;
    RaycastHit[] raycastHitResults = new RaycastHit[1];


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

        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToBody = body.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToBody);

        // if (bp.joint)
        // {
        //     AddVectorObs(GetJointRotation(bp.joint)); // Get the joint rotation
        // }
        if(bp.rb.transform != body)
        {
            AddVectorObs(Quaternion.FromToRotation(body.forward, bp.rb.transform.forward));
        }
    }

    void Update()
    {
		dirToTarget = academy.target.position - bodyParts[body].rb.position;
    }

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
            // print(hit.collider.tag);
            //if it's the ground
            if(hit.collider.CompareTag("ground"))
            {
                dist = hit.distance/maxDist;
                relativeHitPos = body.InverseTransformPoint(hit.point);
            }
        }
        AddVectorObs(dist);
        AddVectorObs(relativeHitPos);
    }



	/// <summary>
    /// Add hit dist to the ground and relative hit position. NonAlloc so it doesn't generate garbage
    /// </summary>
    void RaycastObservationNonAlloc(Vector3 dir, float maxDist)
    {
        float dist = 0;
        Vector3 relativeHitPos = Vector3.zero;
        Ray ray = new Ray(body.position, dir);

        if(Physics.RaycastNonAlloc(ray, raycastHitResults, maxDist, groundLayer, QueryTriggerInteraction.Ignore) > 0)
        {
            // print(raycastHitResults[0].collider.tag);
            dist = raycastHitResults[0].distance/maxDist;
            relativeHitPos = body.InverseTransformPoint(raycastHitResults[0].point);
            // print("dist: " + dist);
            // print("relHitPos: " + relativeHitPos);
        }
        AddVectorObs(dist);
        AddVectorObs(relativeHitPos);
    }


    public override void CollectObservations()
    {
        AddVectorObs(dirToTarget);
        // AddVectorObs(bodyParts[body].rb.rotation);
        // AddVectorObs(Vector3.Dot(dirToTarget.normalized, body.forward)); //are we facing the target?
        // AddVectorObs(Vector3.Dot(bodyParts[body].rb.velocity.normalized, dirToTarget.normalized)); //are we moving towards or away from target?
        
        // Debug.DrawRay(leg0_lower.position, leg0_lower.transform.up, Color.red, 2);
        // Debug.DrawRay(leg1_lower.position, leg1_lower.transform.up, Color.green, 2);
        // Debug.DrawRay(leg2_lower.position, leg2_lower.transform.up, Color.blue, 2);
        // Debug.DrawRay(leg3_lower.position, leg3_lower.transform.up, Color.yellow, 2);
        
        // RaycastObservation(body.position, -body.up, 5);
        // RaycastObservation(body.position, body.forward, 5);

        RaycastObservation(leg0_lower.position, leg0_lower.up, 5);
        RaycastObservation(leg1_lower.position, leg1_lower.up, 5);
        RaycastObservation(leg2_lower.position, leg2_lower.up, 5);
        RaycastObservation(leg3_lower.position, leg3_lower.up, 5);




        // RaycastObservationNonAlloc(body.up, 5);
        // RaycastObservationNonAlloc(-body.up, 5);
        // RaycastObservationNonAlloc(-body.right, 5);
        // RaycastObservationNonAlloc(body.right, 5);
        // RaycastObservationNonAlloc(body.forward, 5);
        // RaycastObservationNonAlloc(-body.forward, 5);
        AddVectorObs(body.forward);
        AddVectorObs(body.up);
        foreach (var bodyPart in bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }

	/// <summary>
    /// Agent touched the target
    /// </summary>
	public void TouchedTarget(float impactForce)
	{
		AddReward(.01f * impactForce); //higher impact should be rewarded
		academy.GetRandomTargetPos();
		Done();
	}

	 public override void AgentAction(float[] vectorAction, string textAction)
    {

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
	
    //Reward moving towards target
    void RewardFunctionMovingTowards()
    {
		movingTowardsDot = Vector3.Dot(bodyParts[body].rb.velocity, dirToTarget.normalized); //don't normalize vel. the faster it goes the more reward it should get
        AddReward(0.03f * movingTowardsDot);
    }

    //Reward facing target
    void RewardFunctionFacingTarget()
    {
		facingDot = Vector3.Dot(dirToTarget.normalized, body.forward);
        AddReward(0.01f * facingDot);
    }

    //Time penalty
    void RewardFunctionTimePenalty()
    {
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
