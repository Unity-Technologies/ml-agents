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
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

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
        jdController.SetupBodyPart(leg0Upper);
        jdController.SetupBodyPart(leg0Lower);
        jdController.SetupBodyPart(leg1Upper);
        jdController.SetupBodyPart(leg1Lower);
        jdController.SetupBodyPart(leg2Upper);
        jdController.SetupBodyPart(leg2Lower);
        jdController.SetupBodyPart(leg3Upper);
        jdController.SetupBodyPart(leg3Lower);
    }

    //We only need to change the joint settings based on decision freq.
    public void IncrementDecisionTimer()
    {
        if(currentDecisionStep == agentParameters.numberOfActionsBetweenDecisions 
           || agentParameters.numberOfActionsBetweenDecisions == 1)
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
        AddReward(1f);
        if(respawnTargetWhenTouched)
        {
		    GetRandomTargetPos();
        }
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
        if (GetReward() < 0f && !IsDone())
        {
            print(GetReward());
        }
        
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
            foot0.material = jdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = jdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = jdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = jdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
        }

        // Joint update logic only needs to happen when a new decision is made
        if(isNewDecisionStep)
        {
            //The dictionary with all the body parts in it are in the jdController
            var bpDict = jdController.bodyPartsDict;

            int i = -1; 
            //pick a new target joint rotation
            bpDict[leg0Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg1Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg2Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg3Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[leg0Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg1Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg2Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[leg3Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);

            //update joint strength
            bpDict[leg0Upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg1Upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg2Upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg3Upper].SetJointStrength(vectorAction[++i]);
            bpDict[leg0Lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg1Lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg2Lower].SetJointStrength(vectorAction[++i]);
            bpDict[leg3Lower].SetJointStrength(vectorAction[++i]);
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
		movingTowardsDot = Vector3.Dot(jdController.bodyPartsDict[body].rb.velocity, dirToTarget.normalized); 
        AddReward(0.03f * movingTowardsDot);
    }

    //Reward facing target & Penalize facing away from target
    void RewardFunctionFacingTarget()
    {
		facingDot = Vector3.Dot(dirToTarget.normalized, body.forward);
        AddReward(0.01f * facingDot);
    }

    //Time penalty - HURRY UP
    void RewardFunctionTimePenalty()
    {
        AddReward(-0.001f); 
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
        
        foreach (var bodyPart in jdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
        isNewDecisionStep = true;
        currentDecisionStep = 1;
    }
}