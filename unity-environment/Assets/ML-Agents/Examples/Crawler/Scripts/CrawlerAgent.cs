using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[RequireComponent(typeof(JointDriveController))] //required to set joint forces
public class CrawlerAgent : Agent {

    [Header("Target To Walk Towards")] 
    [Space(10)] 
    public Transform target;
    public Transform ground;
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
    // public Dictionary<Transform, BodyPart> jdController.bodyParts = new Dictionary<Transform, BodyPart>();
    // public List<BodyPart> jdController.bodyPartsList = new List<BodyPart>(); //to look at values in inspector, just for debugging



    [Header("Joint Settings")] 
    [Space(10)] 
    JointDriveController jdController;
	// public float maxJointSpring;
	// public float jointDampen;
	// public float maxJointForceLimit;
    // // [Tooltip("Reward Functions To Use")] 

    // public float maxJointAngleChangePerDecision; //the change in joint angle will not be able to exceed this value.
    // public float maxJointStrengthChangePerDecision; //the change in joint strenth will not be able to exceed this value.
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
    bool isNewDecisionStep;
    int currentDecisionStep;



    // /// <summary>
    // /// Create BodyPart object and add it to dictionary.
    // /// </summary>
    // public void SetupBodyPart(Transform t)
    // {
    //     BodyPart bp = new BodyPart
    //     {
    //         rb = t.GetComponent<Rigidbody>(),
    //         joint = t.GetComponent<ConfigurableJoint>(),
    //         startingPos = t.position,
    //         startingRot = t.rotation
    //     };
	// 	bp.rb.maxAngularVelocity = 100;
    //     jdController.bodyParts.Add(t, bp);
    //     bp.groundContact = t.GetComponent<GroundContact>();
    //     bp.targetContact = t.GetComponent<TargetContact>();
	// 	// bp.agent = this;
    //     jdController.bodyPartsList.Add(bp);
    // }

    //Initialize
    public override void InitializeAgent()
    {
        jdController = GetComponent<JointDriveController>();
        currentDecisionStep = 1;

        //Setup each body part
        jdController.SetupBodyPart(body);
        jdController.SetupBodyPart(leg0_upper);
        jdController.SetupBodyPart(leg0_lower);
        jdController.SetupBodyPart(leg1_upper);
        jdController.SetupBodyPart(leg1_lower);
        jdController.SetupBodyPart(leg2_upper);
        jdController.SetupBodyPart(leg2_lower);
        jdController.SetupBodyPart(leg3_upper);
        jdController.SetupBodyPart(leg3_lower);

        //we want a lower center of mass or the crawler will roll over easily. 
        //these settings shift the COM on the lower legs
		jdController.bodyParts[leg0_lower].rb.centerOfMass = footCenterOfMassShift;
		jdController.bodyParts[leg1_lower].rb.centerOfMass = footCenterOfMassShift;
		jdController.bodyParts[leg2_lower].rb.centerOfMass = footCenterOfMassShift;
		jdController.bodyParts[leg3_lower].rb.centerOfMass = footCenterOfMassShift;
    }

    //We only need to change the joint settings based on decision freq.
    public void IncrementDecisionTimer()
    {
        if(currentDecisionStep == this.agentParameters.numberOfActionsBetweenDecisions || this.agentParameters.numberOfActionsBetweenDecisions == 1)
        {
            currentDecisionStep = 1;
            isNewDecisionStep = true;
        }
        else
        {
            currentDecisionStep ++;
            isNewDecisionStep = false;
        }
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

        if(bp.rb.transform != body)
        {
            Vector3 localPosRelToBody = body.InverseTransformPoint(rb.position);
            AddVectorObs(localPosRelToBody);
            AddVectorObs(Quaternion.FromToRotation(body.forward, bp.rb.transform.forward));
            // AddVectorObs(Quaternion.FromToRotation(jdController.bodyParts[bp.rb.transform].joint.connectedBody.transform.forward, rb.transform.forward));
        }
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
        jdController.GetCurrentJointForces();
        foreach (var bodyPart in jdController.bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
            // if(!IsDone() && bodyPart.targetContact.touchingTarget)
            // {
            //     TouchedTarget();
            // }
        }
    }


    // void GetCurrentJointForces()
    // {
    //     foreach (var bodyPart in jdController.bodyParts.Values)
    //     {
    //         if(bodyPart.joint)
    //         {
    //             bodyPart.currentJointForce = bodyPart.joint.currentForce;
    //             bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.sqrMagnitude;
    //             bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
    //             bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.sqrMagnitude;
    //         }
    //     }
    // }

	/// <summary>
    /// Agent touched the target
    /// </summary>
	public void TouchedTarget()
	{
		// AddReward(.01f * impactForce); //higher impact should be rewarded
        AddReward(1);
        if(respawnTargetWhenTouched)
        {
		    GetRandomTargetPos();
        }
        print("TouchedTarget()");
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
        print("GetRandomTargetPos()");
        Vector3 newTargetPos = Random.insideUnitSphere * targetSpawnRadius;
		newTargetPos.y = 5;
		target.position = newTargetPos + ground.position;
		// target.position = newTargetPos;
    }




	 public override void AgentAction(float[] vectorAction, string textAction)
    {
        foreach (var bodyPart in jdController.bodyParts.Values)
        {
            if(!IsDone() && bodyPart.targetContact.touchingTarget)
            {
                TouchedTarget();
            }
        }
        //update pos to target
		dirToTarget = target.position - jdController.bodyParts[body].rb.position;

        //if enabled the feet will light up green when the foot is grounded.
        //this is just a visualization and isn't necessary for function
        if(useFootGroundedVisualization)
        {
            foot0.material = jdController.bodyParts[leg0_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot1.material = jdController.bodyParts[leg1_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot2.material = jdController.bodyParts[leg2_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot3.material = jdController.bodyParts[leg3_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
        }

        // Joint update logic only needs to happen when a new decision is made
        if(isNewDecisionStep)
        {
            var bpDict = jdController.bodyParts;

            //pick a new target joint rotation
            bpDict[leg0_upper].SetJointTargetRotation(vectorAction[0], vectorAction[1], 0);
            bpDict[leg1_upper].SetJointTargetRotation(vectorAction[2], vectorAction[3], 0);
            bpDict[leg2_upper].SetJointTargetRotation(vectorAction[4], vectorAction[5], 0);
            bpDict[leg3_upper].SetJointTargetRotation(vectorAction[6], vectorAction[7], 0);
            bpDict[leg0_lower].SetJointTargetRotation(vectorAction[8], 0, 0);
            bpDict[leg1_lower].SetJointTargetRotation(vectorAction[9], 0, 0);
            bpDict[leg2_lower].SetJointTargetRotation(vectorAction[10], 0, 0);
            bpDict[leg3_lower].SetJointTargetRotation(vectorAction[11], 0, 0);

            //update joint strength
            bpDict[leg0_upper].SetJointStrength(vectorAction[12]);
            bpDict[leg1_upper].SetJointStrength(vectorAction[13]);
            bpDict[leg2_upper].SetJointStrength(vectorAction[14]);
            bpDict[leg3_upper].SetJointStrength(vectorAction[15]);
            bpDict[leg0_lower].SetJointStrength(vectorAction[16]);
            bpDict[leg1_lower].SetJointStrength(vectorAction[17]);
            bpDict[leg2_lower].SetJointStrength(vectorAction[18]);
            bpDict[leg3_lower].SetJointStrength(vectorAction[19]);
        }


        // Set reward for this step according to mixture of the following elements.
        if(rewardMovingTowardsTarget){RewardFunctionMovingTowards();}
        // if(rewardFacingTarget){RewardFunctionFacingTarget();}
        if(rewardUseTimePenalty){RewardFunctionTimePenalty();}
        IncrementDecisionTimer();

    }
	
    // //Reward moving towards target & Penalize moving away from target.
    // void RewardFunctionMovingTowards()
    // {
    //     //don't normalize vel. the faster it goes the more reward it should get
    //     //0.03f chosen via experimentation
	// 	movingTowardsDot = Vector3.Dot(jdController.bodyParts[body].rb.velocity, dirToTarget.normalized); 
    //     AddReward(0.03f * movingTowardsDot);
    // }
    //Reward moving towards target & Penalize moving away from target.
    void RewardFunctionMovingTowards()
    {
        //don't normalize vel. the faster it goes the more reward it should get
        //0.03f chosen via experimentation
		// movingTowardsDot = Vector3.Dot(jdController.bodyParts[body].rb.velocity.normalized, dirToTarget.normalized); 
		movingTowardsDot = Vector3.Dot(jdController.bodyParts[body].rb.velocity, dirToTarget.normalized); 
        // movingTowardsDot = Mathf.Clamp(movingTowardsDot, -5, 50f);
        // movingTowardsDot = Mathf.Clamp(movingTowardsDot, -5, 50f);

        // AddReward(0.0003f * movingTowardsDot);
        // moveTowardsReward += 0.01f * movingTowardsDot;
        // moveTowardsReward += 0.003f * movingTowardsDot;
        // totalReward += moveTowardsReward;
        // AddReward(0.01f * movingTowardsDot);
        AddReward(0.03f * movingTowardsDot);
        // AddReward(0.005f * movingTowardsDot);
        // AddReward(0.003f * movingTowardsDot);
        // AddReward(0.03f * movingTowardsDot);

        if(rewardFacingTarget)
        {
            // movingTowardsDot = Vector3.Dot(jdController.bodyParts[body].rb.velocity, dirToTarget.normalized); 
            facingDot = Vector3.Dot(dirToTarget.normalized, body.forward); //up is local forward because capsule is rotated
            // if(movingTowardsDot > .8f)
            if(movingTowardsDot > 0)
            {
                facingDot = Mathf.Clamp(facingDot, 0, 1f);
                // facingReward += 0.001f * facingDot;
                // totalReward += facingReward;
                AddReward(0.01f * facingDot);
            }

        }
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
        print("AgentReset()");
        if(dirToTarget != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(dirToTarget);
        }
        
        foreach (var bodyPart in jdController.bodyParts.Values)
        {
            // bodyPart.Reset();
            bodyPart.Reset(bodyPart);
        }
        isNewDecisionStep = true;
        currentDecisionStep = 1;
    }
}