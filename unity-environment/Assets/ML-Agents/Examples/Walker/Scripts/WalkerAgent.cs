using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(JointDriveController))] //required to set joint forces
public class WalkerAgent : Agent
{
    [Header("Target To Walk Towards")] 
    [Space(10)] 
    public Transform target;
    // public Transform ground;
    // public bool respawnTargetWhenTouched;
    // public float targetSpawnRadius;
	Vector3 dirToTarget;


    [Header("Specific to Walker")] 
    public Vector3 goalDirection;
    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;
    // public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();
    JointDriveController jdController;
    bool isNewDecisionStep;
    int currentDecisionStep;

    float movingTowardsDot;
    float facingDot;
    public LayerMask groundLayer;
    // /// <summary>
    // /// Used to store relevant information for acting and learning for each body part in agent.
    // /// </summary>
    // [System.Serializable]
    // public class BodyPart
    // {
    //     public ConfigurableJoint joint;
    //     public Rigidbody rb;
    //     public Vector3 startingPos;
    //     public Quaternion startingRot;
    //     public GroundContact groundContact;

    //     /// <summary>
    //     /// Reset body part to initial configuration.
    //     /// </summary>
    //     public void Reset()
    //     {
    //         rb.transform.position = startingPos;
    //         rb.transform.rotation = startingRot;
    //         rb.velocity = Vector3.zero;
    //         rb.angularVelocity = Vector3.zero;
    //     }
        
    //     /// <summary>
    //     /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
    //     /// </summary>
    //     public void SetNormalizedTargetRotation(float x, float y, float z, float strength)
    //     {
    //         // Transform values from [-1, 1] to [0, 1]
    //         x = (x + 1f) * 0.5f;
    //         y = (y + 1f) * 0.5f;
    //         z = (z + 1f) * 0.5f;

    //         var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
    //         var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
    //         var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

    //         joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
    //         var jd = new JointDrive
    //         {
    //             positionSpring = ((strength + 1f) * 0.5f) * 10000f,
    //             maximumForce = 250000f
    //         };
    //         joint.slerpDrive = jd;
    //     }

    // }

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
    //     bodyParts.Add(t, bp);
    //     bp.groundContact = t.GetComponent<GroundContact>();
    // }

    public override void InitializeAgent()
    {
        jdController = GetComponent<JointDriveController>();
        jdController.SetupBodyPart(hips);
        jdController.SetupBodyPart(chest);
        jdController.SetupBodyPart(spine);
        jdController.SetupBodyPart(head);
        jdController.SetupBodyPart(thighL);
        jdController.SetupBodyPart(shinL);
        jdController.SetupBodyPart(footL);
        jdController.SetupBodyPart(thighR);
        jdController.SetupBodyPart(shinR);
        jdController.SetupBodyPart(footR);
        jdController.SetupBodyPart(armL);
        jdController.SetupBodyPart(forearmL);
        jdController.SetupBodyPart(handL);
        jdController.SetupBodyPart(armR);
        jdController.SetupBodyPart(forearmR);
        jdController.SetupBodyPart(handR);
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

        // AddVectorObs(Quaternion.Inverse(bp.startingRot) * bp.rb.rotation);
        // AddVectorObs(jdController.bodyParts[hips].rb.position - jdController.bodyParts[hips].startingPos);

            Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
            AddVectorObs(localPosRelToHips);


        // if (bp.joint && (bp.rb.transform != handL && bp.rb.transform != handR))
        // {
        //     var jointRotation = GetJointRotation(bp.joint);
        //     AddVectorObs(jointRotation); // Get the joint rotation
        //     AddVectorObs(bp.previousStrengthValue/jdController.maxJointForceLimit);
        //     // AddVectorObs(bp.currentJointForce);
        //     // AddVectorObs(bp.currentJointTorque);

        // }


        // if(bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        // {
        //     AddVectorObs(Quaternion.Inverse(bp.startingRot) * bp.rb.rotation);

        // }
        if(bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR && bp.rb.transform != footL && bp.rb.transform != footR && bp.rb.transform != head)
        {
            AddVectorObs(Quaternion.FromToRotation(hips.forward, bp.rb.transform.forward));
            AddVectorObs(bp.previousStrengthValue/jdController.maxJointForceLimit);
            // AddVectorObs(Quaternion.FromToRotation(hips.forward, bp.rb.transform.rotation.eulerAngles));
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
                relativeHitPos = hips.InverseTransformPoint(hit.point); 
            }
        }

        //add our raycast observation 
        AddVectorObs(dist);
        AddVectorObs(relativeHitPos);
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations()
    {
        jdController.GetCurrentJointForces();

        AddVectorObs(dirToTarget.normalized);

        //raycast out of the bottom of the feet to get information about where the ground is
        // RaycastObservation(jdController.bodyParts[footL].rb.position, Vector3.down, 3);
        // RaycastObservation(jdController.bodyParts[footR].rb.position, Vector3.down, 3);
        // RaycastObservation(jdController.bodyParts[head].rb.position, Vector3.down, 5);
        // RaycastObservation(jdController.bodyParts[hips].rb.position, Vector3.down, 5);
        // RaycastObservation(jdController.bodyParts[footL].rb.position, -footL.up, 2);
        // RaycastObservation(jdController.bodyParts[footR].rb.position, -footR.up, 2);
       
        //forward & up to help with orientation
        // AddVectorObs(hips.rotation);
        AddVectorObs(jdController.bodyParts[hips].rb.position.y);
        // AddVectorObs(jdController.bodyParts[chest].rb.position.y);
        // AddVectorObs(Vector3.Dot(dirToTarget.normalized, hips.forward));
        // AddVectorObs(Quaternion.Inverse(jdController.bodyParts[hips].startingRot) * jdController.bodyParts[hips].rb.rotation);
        // AddVectorObs(jdController.bodyParts[hips].rb.position - jdController.bodyParts[hips].startingPos);
        // AddVectorObs(Quaternion.FromToRotation(jdController.bodyParts[hips].startingRot, bp.rb.transform.rotation.eulerAngles));

        // AddVectorObs(chest.up);
        AddVectorObs(hips.forward);
        // AddVectorObs(chest.forward);
        AddVectorObs(hips.up);

        foreach (var bodyPart in jdController.bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
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

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        // jdController.GetCurrentJointForces();

        //update pos to target
		dirToTarget = target.position - jdController.bodyParts[hips].rb.position;
        
        
        // Apply action to all relevant body parts. 
        // Joint update logic only needs to happen when a new decision is made
        if(isNewDecisionStep)
        {
            var bpDict = jdController.bodyParts;
            int i = -1;

            //++i syntax is so we don't have to renumber each index when we ant to change the action space
            bpDict[chest].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[spine].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[thighL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[shinL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            // bpDict[footL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[thighR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[shinR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            // bpDict[footR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[armL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[forearmL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[armR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[forearmR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            // bpDict[head].SetJointTargetRotation(vectorAction[++i], 0, 0);
        
        
            //update joint strength settings
            bpDict[chest].SetJointStrength(vectorAction[++i]);
            bpDict[spine].SetJointStrength(vectorAction[++i]);
            // bpDict[head].SetJointStrength(vectorAction[++i]);
            bpDict[thighL].SetJointStrength(vectorAction[++i]);
            bpDict[shinL].SetJointStrength(vectorAction[++i]);
            // bpDict[footL].SetJointStrength(vectorAction[++i]);
            bpDict[thighR].SetJointStrength(vectorAction[++i]);
            bpDict[shinR].SetJointStrength(vectorAction[++i]);
            // bpDict[footR].SetJointStrength(vectorAction[++i]);
            bpDict[armL].SetJointStrength(vectorAction[++i]);
            bpDict[forearmL].SetJointStrength(vectorAction[++i]);
            bpDict[armR].SetJointStrength(vectorAction[++i]);
            bpDict[forearmR].SetJointStrength(vectorAction[++i]);


            // bpDict[chest].SetJointTargetRotation(vectorAction[0], vectorAction[1], vectorAction[2]);
            // bpDict[spine].SetJointTargetRotation(vectorAction[3], vectorAction[4], vectorAction[5]);
            // bpDict[thighL].SetJointTargetRotation(vectorAction[6], vectorAction[7], 0);
            // bpDict[shinL].SetJointTargetRotation(vectorAction[8], 0, 0);
            // bpDict[footL].SetJointTargetRotation(vectorAction[9], vectorAction[10], vectorAction[11]);
            // bpDict[thighR].SetJointTargetRotation(vectorAction[12], vectorAction[13], 0);
            // bpDict[shinR].SetJointTargetRotation(vectorAction[14], 0, 0);
            // bpDict[footR].SetJointTargetRotation(vectorAction[15], vectorAction[16], vectorAction[17]);
            // bpDict[armL].SetJointTargetRotation(vectorAction[18], vectorAction[19], 0);
            // bpDict[forearmL].SetJointTargetRotation(vectorAction[20], 0, 0);
            // bpDict[armR].SetJointTargetRotation(vectorAction[21], vectorAction[22], 0);
            // bpDict[forearmR].SetJointTargetRotation(vectorAction[23], 0, 0);
            // bpDict[head].SetJointTargetRotation(vectorAction[24], vectorAction[25], 0);
        
        
            // //update joint strength settings
            // bpDict[chest].SetJointStrength(vectorAction[26]);
            // bpDict[spine].SetJointStrength(vectorAction[27]);
            // bpDict[head].SetJointStrength(vectorAction[28]);
            // bpDict[thighL].SetJointStrength(vectorAction[29]);
            // bpDict[shinL].SetJointStrength(vectorAction[30]);
            // bpDict[footL].SetJointStrength(vectorAction[31]);
            // bpDict[thighR].SetJointStrength(vectorAction[32]);
            // bpDict[shinR].SetJointStrength(vectorAction[33]);
            // bpDict[footR].SetJointStrength(vectorAction[34]);
            // bpDict[armL].SetJointStrength(vectorAction[35]);
            // bpDict[forearmL].SetJointStrength(vectorAction[36]);
            // bpDict[armR].SetJointStrength(vectorAction[37]);
            // bpDict[forearmR].SetJointStrength(vectorAction[38]);
        }
        




        // bodyParts[chest].SetNormalizedTargetRotation(vectorAction[0], vectorAction[1], vectorAction[2],
        //     vectorAction[26]);
        // bodyParts[spine].SetNormalizedTargetRotation(vectorAction[3], vectorAction[4], vectorAction[5],
        //     vectorAction[27]);

        // bodyParts[thighL].SetNormalizedTargetRotation(vectorAction[6], vectorAction[7], 0, vectorAction[28]);
        // bodyParts[shinL].SetNormalizedTargetRotation(vectorAction[8], 0, 0, vectorAction[29]);
        // bodyParts[footL].SetNormalizedTargetRotation(vectorAction[9], vectorAction[10], vectorAction[11],
        //     vectorAction[30]);
        
        // bodyParts[thighR].SetNormalizedTargetRotation(vectorAction[12], vectorAction[13], 0, vectorAction[31]);
        // bodyParts[shinR].SetNormalizedTargetRotation(vectorAction[14], 0, 0, vectorAction[32]);
        // bodyParts[footR].SetNormalizedTargetRotation(vectorAction[15], vectorAction[16], vectorAction[17],
        //     vectorAction[33]);

        // bodyParts[armL].SetNormalizedTargetRotation(vectorAction[18], vectorAction[19], 0, vectorAction[34]);
        // bodyParts[forearmL].SetNormalizedTargetRotation(vectorAction[20], 0, 0, vectorAction[34]);
        
        // bodyParts[armR].SetNormalizedTargetRotation(vectorAction[21], vectorAction[22], 0, vectorAction[36]);
        // bodyParts[forearmR].SetNormalizedTargetRotation(vectorAction[23], 0, 0, vectorAction[37]);
        
        // bodyParts[head].SetNormalizedTargetRotation(vectorAction[24], vectorAction[25], 0, vectorAction[38]);

        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        // d. Discourage head movement.
        AddReward(
            // + 0.03f * Vector3.Dot(dirToTarget.normalized, jdController.bodyParts[hips].rb.velocity)
            // + 0.01f * Vector3.Dot(dirToTarget.normalized, hips.forward)
            + 0.01f * Vector3.Dot(Vector3.up, head.up)
            + 0.01f * (head.position.y - hips.position.y)
            // - 0.01f * Vector3.Distance(jdController.bodyParts[head].rb.velocity, jdController.bodyParts[hips].rb.velocity)
        );
        RewardFunctionMovingTowards();
        IncrementDecisionTimer();

    }

    // void RewardFunctionMovingTowards()
    // {
	// 	movingTowardsDot = Vector3.Dot(jdController.bodyParts[hips].rb.velocity, dirToTarget.normalized); 
    //     movingTowardsDot = Mathf.Clamp(movingTowardsDot, 0, 50f);
    //     AddReward(0.03f * movingTowardsDot);

    //     // if(rewardFacingTarget)
    //     // {
    //         facingDot = Vector3.Dot(dirToTarget.normalized, hips.forward); //up is local forward because capsule is rotated
    //         if(movingTowardsDot > 0)
    //         {
    //             facingDot = Mathf.Clamp(facingDot, 0, 1f);
    //             AddReward(0.01f * facingDot);
    //         }

    //     // }
    // }
    void RewardFunctionMovingTowards()
    {

        // if(rewardFacingTarget)
        // {
            facingDot = Vector3.Dot(dirToTarget.normalized, hips.forward); //up is local forward because capsule is rotated
            facingDot = Mathf.Clamp(facingDot, 0, 1f);
            AddReward(0.01f * facingDot);
            if(facingDot > 0.5)
            {
                movingTowardsDot = Vector3.Dot(jdController.bodyParts[hips].rb.velocity, dirToTarget.normalized); 
                movingTowardsDot = Mathf.Clamp(movingTowardsDot, 0, 50f);
                AddReward(0.03f * movingTowardsDot);
            }

        // }
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
