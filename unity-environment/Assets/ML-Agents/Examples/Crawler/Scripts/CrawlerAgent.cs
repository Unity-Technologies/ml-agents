using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

[RequireComponent(typeof(JointDriveController))] //required to set joint forces
public class CrawlerAgent : Agent {

    [Header("Target To Walk Towards")] 
    [Space(10)] 
    public Transform target;
    public Transform ground;
    public bool detectTargets;
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

    [Header("Joint Settings")] 
    [Space(10)] 
    JointDriveController jdController;
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

    //Initialize
    public override void InitializeAgent()
    {
        print("InitializeAgent()");
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
            AddVectorObs(bp.currentXNormalizedRot); //current x rot
            AddVectorObs(bp.currentYNormalizedRot); //current y rot
            AddVectorObs(bp.currentZNormalizedRot); //current z rot
            AddVectorObs(bp.currentStrength/jdController.maxJointForceLimit); //curre
        }
    }


    public override void CollectObservations()
    {
        jdController.GetCurrentJointForces();
        //normalize dir vector to help generalize
        AddVectorObs(dirToTarget.normalized);

        //forward & up to help with orientation
        AddVectorObs(body.transform.position.y);
        AddVectorObs(body.forward);
        AddVectorObs(body.up);
        foreach (var bodyPart in jdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }


	/// <summary>
    /// Agent touched the target
    /// </summary>
	public void TouchedTarget()
	{
        AddReward(1);
        if(respawnTargetWhenTouched)
        {
		    GetRandomTargetPos();
        }
		Done();
	}

    /// <summary>
    /// Moves target to a random position within specified radius.
    /// </summary>
    public void GetRandomTargetPos()
    {
        Vector3 newTargetPos = Random.insideUnitSphere * targetSpawnRadius;
		newTargetPos.y = 5;
		target.position = newTargetPos + ground.position;
    }

	 public override void AgentAction(float[] vectorAction, string textAction)
    {
        if(detectTargets)
        {
            foreach (var bodyPart in jdController.bodyPartsDict.Values)
            {
                if(bodyPart.targetContact && !IsDone() && bodyPart.targetContact.touchingTarget)
                {
                    TouchedTarget();
                }
            }
        }

        //update pos to target
		dirToTarget = target.position - jdController.bodyPartsDict[body].rb.position;

        //if enabled the feet will light up green when the foot is grounded.
        //this is just a visualization and isn't necessary for function
        if(useFootGroundedVisualization)
        {
            foot0.material = jdController.bodyPartsDict[leg0_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot1.material = jdController.bodyPartsDict[leg1_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot2.material = jdController.bodyPartsDict[leg2_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
            foot3.material = jdController.bodyPartsDict[leg3_lower].groundContact.touchingGround? groundedMaterial: unGroundedMaterial;
        }

        // Joint update logic only needs to happen when a new decision is made
        if(isNewDecisionStep)
        {
            //The dictionary with all the body parts in it are in the jdController
            var bpDict = jdController.bodyPartsDict;

            int i = -1; 
            //pick a new target joint rotation
            bpDict[leg0_upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg1_upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg2_upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg3_upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg0_lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg1_lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg2_lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg3_lower].SetJointTargetRotation(vectorAction[++i], 0, 0);

            //update joint strength
            bpDict[leg0_upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg1_upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg2_upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg3_upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg0_lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg1_lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg2_lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg3_lower].SetJointStrength(vectorAction[++i]);
        }

        // Set reward for this step according to mixture of the following elements.
        if(rewardMovingTowardsTarget){RewardFunctionMovingTowards();}
        if(rewardFacingTarget){RewardFunctionFacingTarget();}
        if(rewardUseTimePenalty){RewardFunctionTimePenalty();}
        IncrementDecisionTimer();

    }
	
    //Reward moving towards target & Penalize moving away from target.
    void RewardFunctionMovingTowards()
    {
        //don't normalize vel. the faster it goes the more reward it should get
		movingTowardsDot = Vector3.Dot(jdController.bodyPartsDict[body].rb.velocity, dirToTarget.normalized); 
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
        // print("AgentReset()");
        if(dirToTarget != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(dirToTarget);
        }
        
        foreach (var bodyPart in jdController.bodyPartsDict.Values)
        {
            // bodyPart.Reset();
            bodyPart.Reset(bodyPart);
        }
        isNewDecisionStep = true;
        currentDecisionStep = 1;
    }
}