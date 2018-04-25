using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerAgent : Agent
{

    // public float strength;
    // float x_position;
    // [HideInInspector]
    // public bool[] leg_touching;
    // [HideInInspector]
    public bool fell;
    // Vector3 past_velocity;
    // Transform body;
    // Rigidbody bodyRB;
    // public Transform[] limbs;
    // public ConfigurableJoint[] joints;
    // public Rigidbody[] limbRBs;
    // Dictionary<GameObject, Vector3> transformsPosition;
    // Dictionary<GameObject, Quaternion> transformsRotation;
    public float totalCharMass; //total mass of this agent
    public bool visualizeMassDistribution;

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

    public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();
    public bool disableAgentActionsForDebug;
    public float targetJointAngularVelocityScalar; //a scalar for joint.targetAngularVelocity. 100?
    public Vector3 forwardDir;
    // public float actionClampRange = 3; //clamp for our input
    public float maxBodyPartVelocity = 3; //to help tame eratic movement
    public float torquePenalty;
    public float velocityPenalty;
    public float torquePenaltyFinal;
    public float velocityPenaltyFinal;
    public float chestHeightRewardFinal;
    public float chestYUpRewardFinal;
    public float facingDirReward;
        [SerializeField]
        private float[] actionValues;

    // public float agentEnergy = 100;
    // public float energyRegenerationRate;

    [System.Serializable]
    public class BodyPart
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public Vector3 startingPos;
        public Quaternion startingRot;
        public float currentEnergyLevel;
        public WalkerGroundContact groundContactScript;
        // public Quaternion lastTargetJointRotation;
    }

    public void SetupBodyPart(Transform t)
    {
        BodyPart bp = new BodyPart();
        bp.rb = t.GetComponent<Rigidbody>();
        bp.joint = t.GetComponent<ConfigurableJoint>();
        bp.startingPos = t.position;
        bp.startingRot = t.rotation;
        bodyParts.Add(t, bp);
        bp.groundContactScript = t.GetComponent<WalkerGroundContact>();
    }

    public override void InitializeAgent()
    {
        SetupBodyPart(hips);
        SetupBodyPart(chest);
        SetupBodyPart(spine);
        SetupBodyPart(head);
        SetupBodyPart(thighL);
        SetupBodyPart(shinL);
        SetupBodyPart(footL);
        SetupBodyPart(thighR);
        SetupBodyPart(shinR);
        SetupBodyPart(footR);
        SetupBodyPart(armL);
        SetupBodyPart(forearmL);
        SetupBodyPart(handL);
        SetupBodyPart(armR);
        SetupBodyPart(forearmR);
        SetupBodyPart(handR);

        // body = transform.Find("Body");
        // bodyRB = body.GetComponent<Rigidbody>();
        // transformsPosition = new Dictionary<GameObject, Vector3>();
        // transformsRotation = new Dictionary<GameObject, Quaternion>();
        // Transform[] allChildren = GetComponentsInChildren<Transform>();
        // foreach (Transform child in allChildren)
        // {
        //     transformsPosition[child.gameObject] = child.position;
        //     transformsRotation[child.gameObject] = child.rotation;
        // }
        // leg_touching = new bool[2];
        // limbRBs = new Rigidbody[limbs.Length];
        totalCharMass = 0; //reset to 0
        // for (int i = 0; i < limbs.Length; i++)
        // {
        //     limbRBs[i] = limbs[i].gameObject.GetComponent<Rigidbody>();
        //     joints[i] = limbs[i].gameObject.GetComponent<ConfigurableJoint>();
        //     // if(limbRBs[i])
        //     // {
        //     //     limbRBs[i].maxAngularVelocity = 50;
        //     //     limbRBs[i].centerOfMass += limbRBs[i].transform.TransformPoint(joints[i].anchor);
        //     //     totalCharMass += limbRBs[i].mass;
        //     // }
        // }

        foreach(var item in bodyParts)
        {
            if(item.Value.rb)
            {
                  item.Value.rb.maxAngularVelocity = 500;
                // if(joints[i])
                // limbRBs[i].centerOfMass += joints[i].anchor;
                // if(item.Value.joint)
                // {
                //     item.Value.rb.centerOfMass += Vector3.Scale(item.Value.joint.anchor, item.Value.rb.transform.localScale);
                // }
                totalCharMass += item.Value.rb.mass;
            }
        }
        // for (int i = 0; i < limbs.Length; i++)
        // {
        //     // limbRBs[i] = limbs[i].gameObject.GetComponent<Rigidbody>();
        //     // joints[i] = limbs[i].gameObject.GetComponent<ConfigurableJoint>();
        //     if(limbRBs[i])
        //     {
        //         limbRBs[i].maxAngularVelocity = 50;
        //         // if(joints[i])
        //         // limbRBs[i].centerOfMass += joints[i].anchor;
        //         limbRBs[i].centerOfMass += Vector3.Scale(joints[i].anchor, limbRBs[i].transform.localScale);
        //         totalCharMass += limbRBs[i].mass;
        //     }
        // }
    }
    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return(Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    public void BodyPartObservation(BodyPart bp)
        {
            var rb = bp.rb;
            // Vector3 localPosRelToHips = hips.InverseTransformPoint(bp.joint.position); //chilren of the hips are affected by it's scale this is a workaround to get the local pos rel to the hips
            // Vector3 angVelocityRelToHips = hips.InverseTransformVector(rb.angularVelocity); 
            // Vector3 angVelocityRelToConnectedRB = bp.joint.connectedBody.transform.InverseTransformVector(rb.angularVelocity); 
            AddVectorObs(bp.groundContactScript.touchingGround? 1: 0); //is this bp touching the ground
            bp.groundContactScript.touchingGround = false; //reset

            // AddVectorObs(rb.transform.localPosition);
            // AddVectorObs(rb.transform.localRotation.eulerAngles);
            // AddVectorObs(angVelocityRelToHips);
            AddVectorObs(rb.position.y);
            AddVectorObs(rb.velocity);
            AddVectorObs(rb.angularVelocity);

            // if(bp.rb.transform == handL || bp.rb.transform == handR || bp.rb.transform == footL || bp.rb.transform == footR)
            // {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position); //chilren of the hips are affected by it's scale this is a workaround to get the local pos rel to the hips
                AddVectorObs(localPosRelToHips);
                // Vector3 velocityRelToHips = hips.InverseTransformVector(rb.velocity); //
                // AddVectorObs(velocityRelToHips);

            // }
            // else
            // {

            // }
            if(bp.joint)
            {
                if(bp.rb.transform != handL && bp.rb.transform != handR && bp.rb.transform != head)
                {
                    var jointRotation = GetJointRotation(bp.joint);
                    AddVectorObs(jointRotation.eulerAngles); //get the joint rotation
                    // Vector3 angVelocityRelToConnectedRB = bp.joint.connectedBody.transform.InverseTransformVector(rb.angularVelocity); 
                    // AddVectorObs(angVelocityRelToConnectedRB); 
                    AddVectorObs(bp.joint.currentTorque); 

                }

                // AddVectorObs(bp.joint.targetRotation); //get the joint rotation
                // AddVectorObs(bp.lastTargetJointRotation); //get the joint rotation
                
            }
        }
    public override void CollectObservations()
    {

        // AddVectorObs(bodyParts[hips].rb.angularVelocity);
        // AddVectorObs(bodyParts[hips].rb.velocity);
        AddVectorObs(bodyParts[hips].rb.transform.localPosition);
        // AddVectorObs(bodyParts[hips].rb.transform.localRotation);
        AddVectorObs(bodyParts[hips].rb.transform.forward);
        AddVectorObs(bodyParts[hips].rb.transform.up);
        // AddVectorObs(bodyParts[chest].rb.angularVelocity);
        // AddVectorObs(bodyParts[chest].rb.velocity);
        // AddVectorObs(bodyParts[chest].rb.transform.localPosition);
        // AddVectorObs(bodyParts[chest].rb.transform.forward);
        // AddVectorObs(bodyParts[chest].rb.transform.up);
        // AddVectorObs(bodyParts[hips].rb.position);
        // AddVectorObs(bodyParts[chest].rb.transform.forward);
        // AddVectorObs(bodyParts[chest].rb.transform.up);
        // AddVectorObs(FacingTargetDirDot(Vector3.right));  //is our ragdoll facing towards our target dir. we should be able to plug in any vector here
        // AddVectorObs(bodyParts[hips].rb.velocity);
        // AddVectorObs(bodyParts[hips].rb.angularVelocity);
        foreach(var item in bodyParts)
        {
            BodyPartObservation(item.Value);
            // AddVectorObs(item.Value.groundContactScript.touchingGround? 1: 0); //is this bp touching the ground
            // item.Value.groundContactScript.touchingGround = false;
        }

        // for (int index = 0; index < 2; index++)
        // {
        //     // if (leg_touching[index])
        //     // {
        //     //     AddVectorObs(1);
        //     // }
        //     // else
        //     // {
        //     //     AddVectorObs(0);
        //     // }
        //     // if (leg_touching[index])
        //     // {
        //     AddVectorObs(leg_touching[index]? 1: 0);
        //     // }
        //     // else
        //     // {
        //     //     AddVectorObs(0);
        //     // }
        //     // leg_touching[index] = false;
        // }
    }



    // public override void CollectObservations()
    // {

    //     AddVectorObs(bodyParts[hips].rb.rotation.eulerAngles);
    //     // AddVectorObs(bodyParts[hips].rb.velocity);
    //     AddVectorObs(bodyParts[head].rb.position.y); //head height

    //     foreach(var item in bodyParts)
    //     {
    //             var rb = item.Value.rb;
    //             AddVectorObs(item.Key.localPosition);
    //             AddVectorObs(item.Value.rb.position.y);
    //             // AddVectorObs(bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass);
    //             // AddVectorObs(hips.InverseTransformPoint(item.Value.rb.worldCenterOfMass));
    //             // AddVectorObs(bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass);
    //                                     // Gizmos.DrawSphere(item.Value.rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
    //             // AddVectorObs(item.Key.localRotation.eulerAngles);

    //             AddVectorObs(rb.velocity);
    //             AddVectorObs(rb.angularVelocity);


    //         if(item.Key != hips)
    //         {
    //             // AddVectorObs(Quaternion.FromToRotation(hips.transform.forward, item.Key.forward).eulerAngles); //can't parent to hips because it skews model so have to do this instead of local rotation
    //             var jointRotation = GetJointRotation(item.Value.joint);
    //             AddVectorObs(jointRotation.eulerAngles); //get the joint rotation
    //             // print(item.Key.name + " joint rotation: " + jointRotation);
    //         }


    //             // //let ml handle body part mass
    //             // AddVectorObs(rb.mass);

    //         // }
    //     }

    //     for (int index = 0; index < 2; index++)
    //     {
    //         if (leg_touching[index])
    //         {
    //             AddVectorObs(1);
    //         }
    //         else
    //         {
    //             AddVectorObs(0);
    //         }
    //         leg_touching[index] = false;
    //     }
    // }
    // public override void CollectObservations()
    // {
    //     // AddVectorObs(body.transform.rotation);
    //     AddVectorObs(bodyRB.rotation.eulerAngles);
    //     // AddVectorObs(body.transform.rotation.eulerAngles);

    //     AddVectorObs(bodyRB.velocity);
    //     AddVectorObs(limbRBs[8].position.y); //head height
    //     // AddVectorObs(bodyRB.position.y);

    //     //let ml handle body part mass
    //     AddVectorObs(bodyRB.mass);

    //     // AddVectorObs((bodyRB.velocity - past_velocity) / Time.fixedDeltaTime);
    //     // past_velocity = bodyRB.velocity;

    //     for (int i = 0; i < limbs.Length; i++)
    //     {
    //         AddVectorObs(limbs[i].localPosition);
    //         AddVectorObs(limbs[i].localRotation.eulerAngles);
    //         // print("localrotation: " + limbs[i].localRotation.eulerAngles);
    //         // AddVectorObs(limbs[i].localRotation);
    //         AddVectorObs(limbRBs[i].velocity);
    //         AddVectorObs(limbRBs[i].angularVelocity);

    //         AddVectorObs(GetJointRotation(joints[i]).eulerAngles); //get the joint rotation
    //         // print(GetJointRotation(joints[i]).eulerAngles);


    //         //let ml handle body part mass
    //         AddVectorObs(limbRBs[i].mass);
    //     }

    //     for (int index = 0; index < 2; index++)
    //     {
    //         if (leg_touching[index])
    //         {
    //             AddVectorObs(1);
    //         }
    //         else
    //         {
    //             AddVectorObs(0);
    //         }
    //         leg_touching[index] = false;
    //     }
    // }

    // public void SetNormalizedTargetRotation(float x, float y, float z)
    // {
    //     //rigidbody.AddRelativeTorque(x * 15f, y * 15f, z * 15f, ForceMode.Acceleration);
    //     //return;
    //     //x = Mathf.InverseLerp(-1f,1f,x);
    //     x = Mathf.Clamp(x, -1f, 1f);
    //     y = Mathf.Clamp(y, -1f, 1f);
    //     z = Mathf.Clamp(z, -1f, 1f);
    //     y = (y + 1f) * 0.5f;
    //     z = (z + 1f) * 0.5f;

    //     float xRot;

    //     if (x <= 0f)
    //     {
    //         x = x + 1f;
    //         xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, 0f, x);
    //     }
    //     else
    //     {
            
    //         xRot = Mathf.Lerp(0f, joint.highAngularXLimit.limit, x);
    //     }
    //     float yRot = Mathf.Lerp( -joint.angularYLimit.limit, joint.angularYLimit.limit,y);
    //     float zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

    //     joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
    // }
    // public void SetNormalizedTargetRotation(BodyPart bp, float x, float y, float z)
    public void SetNormalizedTargetRotation(BodyPart bp, float x, float y, float z, float posSpring, float posDamper)
    {

        float xRot = 0;
        float yRot = 0;
        float zRot = 0;
        if(x != 0)
        {
            // x = Mathf.Clamp(x, -1f, 1f);
            if (x <= 0f)
            {
                x = x + 1f;
                xRot = Mathf.Lerp(bp.joint.lowAngularXLimit.limit, 0f, x);
            }
            else
            {
                
                xRot = Mathf.Lerp(0f, bp.joint.highAngularXLimit.limit, x);
            }
        }

        if(y != 0)
        {
            // y = Mathf.Clamp(y, -1f, 1f);
            y = (y + 1f) * 0.5f;
            yRot = Mathf.Lerp( -bp.joint.angularYLimit.limit, bp.joint.angularYLimit.limit,y);
        }

        if(z != 0)
        {
            // z = Mathf.Clamp(z, -1f, 1f);
            z = (z + 1f) * 0.5f;
            zRot = Mathf.Lerp(-bp.joint.angularZLimit.limit, bp.joint.angularZLimit.limit, z);
        }

        bp.joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
        JointDrive jd = new JointDrive();
        jd.positionSpring = posSpring * 10000f;
        jd.positionDamper = posDamper * 500f;
        jd.maximumForce = 250000f;
        bp.joint.slerpDrive = jd;
        // jd.mode



        // bp.joint.sl
        // bp.lastTargetJointRotation = bp.joint.targetRotation;
    }

    public void SetJointTargetAngularVelocity(BodyPart bp, float x, float y, float z)
    {
        // Joint solver equation from docs. this is dumb
        // force = spring * (targetPosition - position) + damping * (targetVelocity - velocity)

        float xVel = 0;
        float yVel = 0;
        float zVel = 0;
        // float clampRange = 3;

            // x = Mathf.Clamp(x, -actionClampRange, actionClampRange);
            // y = Mathf.Clamp(y, -actionClampRange, actionClampRange);
            // z = Mathf.Clamp(z, -actionClampRange, actionClampRange);
        xVel = x * targetJointAngularVelocityScalar;
        yVel = y * targetJointAngularVelocityScalar;
        zVel = z * targetJointAngularVelocityScalar;
        bp.joint.targetAngularVelocity = new Vector3(xVel, yVel, zVel);
    }

    //the dot product of two vectors facing exactly the same dir is 1. 
    //if two vectors are facing exactly opposite dir, the dp is -1.
    //a dp of say .9 means two vectors are almost pointing the same way.
    //a dp of 0 means two vectors are perpendicular (i.e. 90 degree diff)
    //we can use DotProduct to determine if the ragdoll is facing our target dir or not
    float FacingTargetDirDot(Vector3 targetDir)
    {
        // bool facingDir = false;
        // float facingDirDot;
        // var targetDir = Vector3.right;
        forwardDir = bodyParts[hips].rb.transform.forward;
        // forwardDir.y = 0;
        // forwardDir.z = 0;
        facingDirReward = Vector3.Dot(forwardDir, targetDir);
        return facingDirReward;
        // if(facingTowardsDot >= .9f)//roughly facing dir
        // {
        //     facingDir = true;
        // }
        // return facingDir;
    }

    bool IsFacingTargetDir()
    {
        bool facingDir = false;
        var targetDir = Vector3.right;
        Vector3 forwardDir = bodyParts[hips].rb.transform.forward;
        forwardDir.y = 0;
        forwardDir.z = 0;
        float facingTowardsDot = Vector3.Dot(forwardDir, targetDir);
        if(facingTowardsDot >= .9f)//roughly facing dir
        {
            facingDir = true;
        }
        return facingDir;
    }


    //SET TARGET ANGULAR VELOCITY OF THE JOINT
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if(!disableAgentActionsForDebug)
        {
            actionValues = vectorAction;
            // continue;
        // }


    // //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], vectorAction[2]);
    // //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[spine], vectorAction[1], 0, 0);
    //    SetJointTargetAngularVelocity(bodyParts[chest], vectorAction[0], vectorAction[1], 0);
    //    SetJointTargetAngularVelocity(bodyParts[spine], vectorAction[2], vectorAction[3], 0);

    //    SetJointTargetAngularVelocity(bodyParts[thighL], vectorAction[4], vectorAction[5], 0);
    //    SetJointTargetAngularVelocity(bodyParts[shinL], vectorAction[6], 0, 0);
    //    SetJointTargetAngularVelocity(bodyParts[footL], vectorAction[7], vectorAction[8], vectorAction[9]);
    //    SetJointTargetAngularVelocity(bodyParts[thighR], vectorAction[10], vectorAction[11], 0);
    //    SetJointTargetAngularVelocity(bodyParts[shinR], vectorAction[12], 0, 0);
    //    SetJointTargetAngularVelocity(bodyParts[footR], vectorAction[13], vectorAction[14], vectorAction[15]);
       
    //    SetJointTargetAngularVelocity(bodyParts[armL], vectorAction[16], vectorAction[17], 0);
    //    SetJointTargetAngularVelocity(bodyParts[forearmL], vectorAction[18], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handL], vectorAction[0], 0, 0);
    //    SetJointTargetAngularVelocity(bodyParts[armR], vectorAction[19], vectorAction[20], 0);
    //    SetJointTargetAngularVelocity(bodyParts[forearmR], vectorAction[21], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handR], vectorAction[0], 0, 0);



    //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], 0);
    //    SetNormalizedTargetRotation(bodyParts[spine], vectorAction[2], vectorAction[3], 0);

    //    SetNormalizedTargetRotation(bodyParts[thighL], vectorAction[4], vectorAction[5], 0);
    //    SetNormalizedTargetRotation(bodyParts[shinL], vectorAction[6], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[footL], vectorAction[7], vectorAction[8], vectorAction[9]);
    //    SetNormalizedTargetRotation(bodyParts[thighR], vectorAction[10], vectorAction[11], 0);
    //    SetNormalizedTargetRotation(bodyParts[shinR], vectorAction[12], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[footR], vectorAction[13], vectorAction[14], vectorAction[15]);
       
    //    SetNormalizedTargetRotation(bodyParts[armL], vectorAction[16], vectorAction[17], 0);
    //    SetNormalizedTargetRotation(bodyParts[forearmL], vectorAction[18], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handL], vectorAction[0], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[armR], vectorAction[19], vectorAction[20], 0);
    //    SetNormalizedTargetRotation(bodyParts[forearmR], vectorAction[21], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handR], vectorAction[0], 0, 0);



       SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], 0, vectorAction[2], vectorAction[3]);
       SetNormalizedTargetRotation(bodyParts[spine], vectorAction[4], vectorAction[5], 0, vectorAction[6], vectorAction[7]);

       SetNormalizedTargetRotation(bodyParts[thighL], vectorAction[8], vectorAction[9], 0, vectorAction[10], vectorAction[11]);
       SetNormalizedTargetRotation(bodyParts[shinL], vectorAction[12], 0, 0, vectorAction[13], vectorAction[14]);
       SetNormalizedTargetRotation(bodyParts[footL], vectorAction[15], vectorAction[16], vectorAction[17], vectorAction[18], vectorAction[19]);
       SetNormalizedTargetRotation(bodyParts[thighR], vectorAction[20], vectorAction[21], 0, vectorAction[22], vectorAction[23]);
       SetNormalizedTargetRotation(bodyParts[shinR], vectorAction[24], 0, 0, vectorAction[25], vectorAction[26]);
       SetNormalizedTargetRotation(bodyParts[footR], vectorAction[27], vectorAction[28], vectorAction[29], vectorAction[30], vectorAction[31]);
       
       SetNormalizedTargetRotation(bodyParts[armL], vectorAction[32], vectorAction[33], 0, vectorAction[34], vectorAction[35]);
       SetNormalizedTargetRotation(bodyParts[forearmL], vectorAction[36], 0, 0, vectorAction[37], vectorAction[38]);
    //    SetNormalizedTargetRotation(bodyParts[handL], vectorAction[0], 0, 0);
       SetNormalizedTargetRotation(bodyParts[armR], vectorAction[39], vectorAction[40], 0, vectorAction[41], vectorAction[42]);
       SetNormalizedTargetRotation(bodyParts[forearmR], vectorAction[43], 0, 0, vectorAction[44], vectorAction[45]);
    //    SetNormalizedTargetRotation(bodyParts[handR], vectorAction[0], 0, 0);





        // for (int k = 0; k < 21; k++)
        // {
        //     // torquePenalty += vectorAction[k] * vectorAction[k];
        //     // torquePenalty +=  Mathf.Abs(Mathf.Clamp(vectorAction[k], -actionClampRange, actionClampRange));
        //     torquePenalty +=  Mathf.Abs(vectorAction[k]);
        //     // torquePenalty +=  Mathf.Abs(vectorAction[k]);
        //     // print(vectorAction[k] * vectorAction[k]);
        // }
        // foreach(var item in bodyParts)
        // {
        // } 
        velocityPenalty = 0; 
        torquePenalty = 0; 
        foreach(var item in bodyParts)
        {
            var velSM = item.Value.rb.velocity.sqrMagnitude;
            if(velSM > maxBodyPartVelocity * maxBodyPartVelocity)
            {
                velocityPenalty += velSM;
            }


            if(item.Value.joint)
            {
                var currentTorqueSM = item.Value.joint.currentTorque.sqrMagnitude ;
                // print(item.Value.rb.name + " joint torque: " + item.Value.joint.currentTorque);
                // print(item.Value.rb.name + " joint torqueSM: " + item.Value.joint.currentTorque.sqrMagnitude);
                if(currentTorqueSM > 25000)
                {
                    torquePenalty += currentTorqueSM/1000; //value determined by experimentation
                }

            }
        } 
            // print("tp: " + 0.001f * torquePenalty);
            // print("chest y pos: " + 0.01f * bodyParts[chest].rb.position.y);

        // AddReward(
        //     - 0.001f * torquePenalty
        //     + 0.01f * bodyParts[chest].rb.position.y
        // );
        
        torquePenaltyFinal = 0.001f * Mathf.Clamp(torquePenalty, 0f, 500);
        velocityPenaltyFinal = 0.001f * Mathf.Clamp(velocityPenalty, 0f, 500);
        chestHeightRewardFinal = 0.2f * bodyParts[chest].rb.position.y;
        chestYUpRewardFinal = 0.2f * Vector3.Dot(bodyParts[chest].rb.transform.up, Vector3.up);
        AddReward(
            // - 0.001f * torquePenalty
            // - 0.001f * velocityPenalty
            // - torquePenaltyFinal
            - velocityPenaltyFinal
            // - 0.001f * Mathf.Clamp(velocityPenalty, 0f, 1000f)
            // - 0.001f * velocityPenalty
            // - 0.005f * Mathf.Abs(bodyParts[chest].rb.velocity.y)
            
            // + 0.001f * facingDirReward //are we facing our target dir? this dir should change when our target dir changes. a FacingTargetDirDot of 1 means our character is facing exactly towards our target dir
            // + 0.02f * Mathf.Clamp(bodyParts[hips].rb.velocity.x, 0f, 1000f)
            // + 0.02f * Mathf.Clamp(bodyParts[chest].rb.position.y, 0f, 100f)
            // + 0.01f * Mathf.Clamp(Vector3.Dot(bodyParts[chest].rb.transform.up, Vector3.up), 0f, 100f) //reward for chest up dir == world up dir
            + chestHeightRewardFinal
            // + bodyParts[chest].rb.position.y
            + chestYUpRewardFinal
        );
        // AddReward(bodyParts[head].rb.velocity.sqrMagnitude * -.001f);
            // SetReward((ragdoll.head.Height - 1.2f) + ragdoll.head.transform.up.y * 0.1f);

            // if (ragdoll.upperChest.touchingGround || ragdoll.lowerChest.touchingGround || ragdoll.head.touchingGround || ragdoll.head.Height < 1.2f)
            // {
            //     SetReward(-1f);
            //     if (Application.isEditor)
            //         print(GetCumulativeReward());
            //     Done();
        }
    }



    //SET TARGET ANGLES DIRECTLY
    // public override void AgentAction(float[] vectorAction, string textAction)
    // {
    //     if(!disableAgentActionsForDebug)
    //     {
    //         // continue;
    //     // }


    // //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], vectorAction[2]);
    // //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[spine], vectorAction[1], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], 0);
    //    SetNormalizedTargetRotation(bodyParts[spine], vectorAction[2], vectorAction[3], 0);

    //    SetNormalizedTargetRotation(bodyParts[thighL], vectorAction[4], vectorAction[5], 0);
    //    SetNormalizedTargetRotation(bodyParts[shinL], vectorAction[6], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[footL], vectorAction[7], vectorAction[8], vectorAction[9]);
    //    SetNormalizedTargetRotation(bodyParts[thighR], vectorAction[10], vectorAction[11], 0);
    //    SetNormalizedTargetRotation(bodyParts[shinR], vectorAction[12], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[footR], vectorAction[13], vectorAction[14], vectorAction[15]);
       
    //    SetNormalizedTargetRotation(bodyParts[armL], vectorAction[16], vectorAction[17], 0);
    //    SetNormalizedTargetRotation(bodyParts[forearmL], vectorAction[18], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handL], vectorAction[0], 0, 0);
    //    SetNormalizedTargetRotation(bodyParts[armR], vectorAction[19], vectorAction[20], 0);
    //    SetNormalizedTargetRotation(bodyParts[forearmR], vectorAction[21], 0, 0);
    // //    SetNormalizedTargetRotation(bodyParts[handR], vectorAction[0], 0, 0);



    //     float torquePenalty = 0; 
    //     for (int k = 0; k < 21; k++)
    //     {
    //         // torquePenalty += vectorAction[k] * vectorAction[k];
    //         torquePenalty +=  Mathf.Abs(vectorAction[k]);
    //         // print(vectorAction[k] * vectorAction[k]);
    //     }
    //         // print("tp: " + 0.001f * torquePenalty);
    //         // print("chest y pos: " + 0.01f * bodyParts[chest].rb.position.y);

    //     // AddReward(
    //     //     - 0.001f * torquePenalty
    //     //     + 0.01f * bodyParts[chest].rb.position.y
    //     // );


    //     AddReward(
    //         // - 0.001f * torquePenalty
    //         + 0.02f * Mathf.Clamp(bodyParts[hips].rb.velocity.x, 0f, 1000f)
    //         + 0.01f * bodyParts[chest].rb.position.y
    //         - 0.005f * Mathf.Abs(bodyParts[chest].rb.velocity.y)
    //     );
    //     // AddReward(bodyParts[head].rb.velocity.sqrMagnitude * -.001f);
    //         // SetReward((ragdoll.head.Height - 1.2f) + ragdoll.head.transform.up.y * 0.1f);

    //         // if (ragdoll.upperChest.touchingGround || ragdoll.lowerChest.touchingGround || ragdoll.head.touchingGround || ragdoll.head.Height < 1.2f)
    //         // {
    //         //     SetReward(-1f);
    //         //     if (Application.isEditor)
    //         //         print(GetCumulativeReward());
    //         //     Done();
    //     }
    // }



    
    // public override void AgentAction(float[] vectorAction, string textAction)
    // {
    //     for (int k = 0; k < vectorAction.Length; k++)
    //     {
    //         vectorAction[k] = Mathf.Clamp(vectorAction[k], -1f, 1f);
    //     }
    //     // ForceMode forceModeToUse = ForceMode.VelocityChange;
    //     ForceMode forceModeToUse = ForceMode.Acceleration;
    //     // ForceMode forceModeToUse = ForceMode.Force;

    //     bodyParts[thighL].rb.AddTorque(thighL.right * strength * vectorAction[0], forceModeToUse);
    //     bodyParts[thighR].rb.AddTorque(thighR.right * strength * vectorAction[1], forceModeToUse);
    //     bodyParts[thighL].rb.AddTorque(thighL.forward * strength * vectorAction[2], forceModeToUse);
    //     bodyParts[thighR].rb.AddTorque(thighR.forward * strength * vectorAction[3], forceModeToUse);
    //     bodyParts[shinL].rb.AddTorque(shinL.right * strength * vectorAction[4], forceModeToUse);
    //     bodyParts[shinR].rb.AddTorque(shinR.right * strength * vectorAction[5], forceModeToUse);
    //     // bodyParts[spine].rb.AddTorque(spine.up * strength * vectorAction[6], forceModeToUse);
    //     // bodyParts[spine].rb.AddTorque(spine.forward * strength * vectorAction[7], forceModeToUse);
    //     bodyParts[chest].rb.AddTorque(chest.up * strength * vectorAction[6], forceModeToUse);
    //     bodyParts[chest].rb.AddTorque(chest.forward * strength * vectorAction[7], forceModeToUse);
    //     // bodyParts[head].rb.AddTorque(head.up * strength * vectorAction[10], forceModeToUse);
    //     // bodyParts[head].rb.AddTorque(head.forward * strength * vectorAction[11], forceModeToUse);
    //     bodyParts[armL].rb.AddTorque(armL.forward * strength * vectorAction[8], forceModeToUse);
    //     bodyParts[armL].rb.AddTorque(armL.right * strength * vectorAction[9], forceModeToUse);
    //     bodyParts[armR].rb.AddTorque(armR.forward * strength * vectorAction[10], forceModeToUse);
    //     bodyParts[armR].rb.AddTorque(armR.right * strength * vectorAction[11], forceModeToUse);
    //     bodyParts[forearmR].rb.AddTorque(forearmR.right * strength * vectorAction[12], forceModeToUse);
    //     bodyParts[forearmL].rb.AddTorque(forearmL.right * strength * vectorAction[13], forceModeToUse);

    //     // bodyParts[thighL].rb.AddTorque(thighL.right * strength * vectorAction[0], forceModeToUse);
    //     // bodyParts[thighR].rb.AddTorque(thighR.right * strength * vectorAction[1], forceModeToUse);
    //     // bodyParts[thighL].rb.AddTorque(thighL.forward * strength * vectorAction[2], forceModeToUse);
    //     // bodyParts[thighR].rb.AddTorque(thighR.forward * strength * vectorAction[3], forceModeToUse);
    //     // bodyParts[shinL].rb.AddTorque(shinL.right * strength * vectorAction[4], forceModeToUse);
    //     // bodyParts[shinR].rb.AddTorque(shinR.right * strength * vectorAction[5], forceModeToUse);
    //     // // bodyParts[spine].rb.AddTorque(spine.up * strength * vectorAction[6], forceModeToUse);
    //     // // bodyParts[spine].rb.AddTorque(spine.forward * strength * vectorAction[7], forceModeToUse);
    //     // bodyParts[chest].rb.AddTorque(chest.up * strength * vectorAction[8], forceModeToUse);
    //     // bodyParts[chest].rb.AddTorque(chest.forward * strength * vectorAction[9], forceModeToUse);
    //     // // bodyParts[head].rb.AddTorque(head.up * strength * vectorAction[10], forceModeToUse);
    //     // // bodyParts[head].rb.AddTorque(head.forward * strength * vectorAction[11], forceModeToUse);
    //     // bodyParts[armL].rb.AddTorque(armL.forward * strength * vectorAction[12], forceModeToUse);
    //     // bodyParts[armL].rb.AddTorque(armL.right * strength * vectorAction[13], forceModeToUse);
    //     // bodyParts[armR].rb.AddTorque(armR.forward * strength * vectorAction[14], forceModeToUse);
    //     // bodyParts[armR].rb.AddTorque(armR.right * strength * vectorAction[15], forceModeToUse);
    //     // bodyParts[forearmR].rb.AddTorque(forearmR.right * strength * vectorAction[16], forceModeToUse);
    //     // bodyParts[forearmL].rb.AddTorque(forearmL.right * strength * vectorAction[17], forceModeToUse);

    //     // float torquePenalty = 0; 
    //     // for (int k = 0; k < 17; k++)
    //     // {
    //     //     torquePenalty += vectorAction[k] * vectorAction[k];
    //     // }
    //     float torquePenalty = 0; 
    //     for (int k = 0; k < 13; k++)
    //     {
    //         torquePenalty += vectorAction[k] * vectorAction[k];
    //     }
    //     float velocityPenalty = 0; 
    //     foreach(var item in bodyParts)
    //     {
            
    //         if(item.Key != hips)
    //         {
    //             velocityPenalty += item.Value.rb.velocity.sqrMagnitude;
    //         }
    //     }






    //     // //let ml handle body part mass
    //     // int actIndex = 18;
    //     // foreach(var item in bodyParts)
    //     // {
    //     //     item.Value.rb.mass = Mathf.Clamp(vectorAction[actIndex], 0.1f, 1f) * 20;
    //     //     actIndex++;
    //     // }



    //     if (!IsDone())
    //     {
    //         // float headHeightReward = bodyParts[head].rb.position.y > 5? 1.0f * bodyParts[head].rb.position.y: 0;
    //         float headHeightReward = bodyParts[head].rb.position.y/2;
    //         float hipsHeightReward = bodyParts[hips].rb.position.y/2;
    //         // SetReward(
    //         AddReward(
    //         - 0.05f * torquePenalty 
    //         // - 0.01f * velocityPenalty
    //         // + .5f * limbRBs[8].velocity.x
    //         + .5f * bodyParts[hips].rb.velocity.x
    //         // + 1.0f * bodyRB.velocity.x
    //         // + 1.0f * bodyParts[head].rb.position.y //head height
    //         + headHeightReward //head height
    //         + hipsHeightReward //head height
    //         // + 1f * bodyRB.position.y
    //         - 0.05f * Mathf.Abs(hips.transform.position.z - hips.transform.parent.transform.position.z)
    //         // - 0.05f * Mathf.Abs(bodyRB.velocity.y)
    //         - 0.05f * Mathf.Abs(bodyParts[hips].rb.angularVelocity.sqrMagnitude)
    //         );
            
    //     }
    //     if (fell)
    //     {
    //         Done();
    //         AddReward(-1f);
    //     }
    // }



    // public override void AgentAction(float[] vectorAction, string textAction)
    // {
    //     for (int k = 0; k < vectorAction.Length; k++)
    //     {
    //         vectorAction[k] = Mathf.Clamp(vectorAction[k], -1f, 1f);
    //     }

    //     limbRBs[0].AddTorque(-limbs[0].transform.right * strength * vectorAction[0], ForceMode.Force);
    //     limbRBs[1].AddTorque(-limbs[1].transform.right * strength * vectorAction[1], ForceMode.Force);
    //     // limbRBs[2].AddTorque(-limbs[2].transform.right * strength * vectorAction[2], ForceMode.VelocityChange);
    //     // limbRBs[3].AddTorque(-limbs[3].transform.right * strength * vectorAction[3], ForceMode.VelocityChange);
    //     // limbRBs[0].AddTorque(-limbs[0].transform.forward * strength * vectorAction[4]);
    //     // limbRBs[1].AddTorque(-limbs[1].transform.forward * strength * vectorAction[5]);
    //     // limbRBs[2].AddTorque(-limbs[2].transform.forward * strength * vectorAction[6]);
    //     // limbRBs[3].AddTorque(-limbs[3].transform.forward * strength * vectorAction[7]);
    //     // limbRBs[0].AddTorque(-limbs[0].transform.forward * strength * vectorAction[2], ForceMode.VelocityChange);
    //     // limbRBs[1].AddTorque(-limbs[1].transform.forward * strength * vectorAction[3], ForceMode.VelocityChange);
    //     limbRBs[0].AddTorque(-body.transform.up * strength * vectorAction[2], ForceMode.VelocityChange);
    //     limbRBs[1].AddTorque(-body.transform.up * strength * vectorAction[3], ForceMode.VelocityChange);
    //     // limbRBs[2].AddTorque(-body.transform.up * strength * vectorAction[6], ForceMode.VelocityChange);
    //     // limbRBs[3].AddTorque(-body.transform.up * strength * vectorAction[7], ForceMode.VelocityChange);
    //     limbRBs[2].AddTorque(-limbs[2].transform.right * strength * vectorAction[4], ForceMode.VelocityChange);
    //     limbRBs[3].AddTorque(-limbs[3].transform.right * strength * vectorAction[5], ForceMode.VelocityChange);


    //     limbRBs[6].AddTorque(-limbs[6].transform.forward * strength * vectorAction[6], ForceMode.VelocityChange);
    //     limbRBs[7].AddTorque(-limbs[7].transform.forward * strength * vectorAction[7], ForceMode.VelocityChange);
    //     limbRBs[8].AddTorque(-limbs[8].transform.forward * strength * vectorAction[8], ForceMode.VelocityChange);
    //     limbRBs[6].AddTorque(-limbs[6].transform.up * strength * vectorAction[9], ForceMode.VelocityChange);
    //     limbRBs[7].AddTorque(-limbs[7].transform.up * strength * vectorAction[10], ForceMode.VelocityChange);
    //     limbRBs[8].AddTorque(-limbs[8].transform.up * strength * vectorAction[11], ForceMode.VelocityChange);
    //     // limbRBs[4].AddTorque(-limbs[4].transform.up * strength * vectorAction[7], ForceMode.VelocityChange);
    //     // limbRBs[6].AddTorque(-limbs[6].transform.right * strength * vectorAction[10], ForceMode.VelocityChange);
    //     // limbRBs[7].AddTorque(-limbs[7].transform.right * strength * vectorAction[11], ForceMode.VelocityChange);
        
    //     // limbRBs[0].AddTorque(-limbs[0].transform.right * strength * vectorAction[0]);
    //     // limbRBs[1].AddTorque(-limbs[1].transform.right * strength * vectorAction[1]);
    //     // limbRBs[2].AddTorque(-limbs[2].transform.right * strength * vectorAction[2]);
    //     // limbRBs[3].AddTorque(-limbs[3].transform.right * strength * vectorAction[3]);
    //     // // limbRBs[0].AddTorque(-limbs[0].transform.forward * strength * vectorAction[4]);
    //     // // limbRBs[1].AddTorque(-limbs[1].transform.forward * strength * vectorAction[5]);
    //     // // limbRBs[2].AddTorque(-limbs[2].transform.forward * strength * vectorAction[6]);
    //     // // limbRBs[3].AddTorque(-limbs[3].transform.forward * strength * vectorAction[7]);
    //     // limbRBs[0].AddTorque(-body.transform.up * strength * vectorAction[4]);
    //     // limbRBs[1].AddTorque(-body.transform.up * strength * vectorAction[5]);
    //     // limbRBs[2].AddTorque(-body.transform.up * strength * vectorAction[6]);
    //     // limbRBs[3].AddTorque(-body.transform.up * strength * vectorAction[7]);
    //     // limbRBs[4].AddTorque(-limbs[4].transform.right * strength * vectorAction[8]);
    //     // limbRBs[5].AddTorque(-limbs[5].transform.right * strength * vectorAction[9]);
    //     // limbRBs[6].AddTorque(-limbs[6].transform.right * strength * vectorAction[10]);
    //     // limbRBs[7].AddTorque(-limbs[7].transform.right * strength * vectorAction[11]);





    //     //let ml handle body part mass
    //     int actIndex = 12;
    //     for (int i = 0; i < limbRBs.Length; i++)
    //     {
    //         limbRBs[i].mass = Mathf.Clamp(vectorAction[actIndex], 0.1f, 1f) * 20;
    //         actIndex++;
    //     }
    //     bodyRB.mass = Mathf.Clamp(vectorAction[21], 0.1f, 1f) * 20;






    //     float torque_penalty = vectorAction[0] * vectorAction[0] + 
    //         vectorAction[1] * vectorAction[1] + 
    //         vectorAction[2] * vectorAction[2] + 
    //         vectorAction[3] * vectorAction[3] +
    //         vectorAction[4] * vectorAction[4] + 
    //         vectorAction[5] * vectorAction[5] +
    //         vectorAction[6] * vectorAction[6] +
    //         vectorAction[7] * vectorAction[7] +
    //         vectorAction[8] * vectorAction[8] + 
    //         vectorAction[9] * vectorAction[9] + 
    //         vectorAction[10] * vectorAction[10] + 
    //         vectorAction[11] * vectorAction[11]
    //         ;

    //     if (!IsDone())
    //     {
    //         SetReward(
    //         0 - 0.01f * torque_penalty 
    //         // + .5f * limbRBs[8].velocity.x
    //         + .5f * bodyRB.velocity.x
    //         // + 1.0f * bodyRB.velocity.x
    //         + 1.0f * limbRBs[8].position.y //head height
    //         // + 1f * bodyRB.position.y
    //         - 0.05f * Mathf.Abs(body.transform.position.z - body.transform.parent.transform.position.z)
    //         // - 0.05f * Mathf.Abs(bodyRB.velocity.y)
    //         - 0.05f * Mathf.Abs(bodyRB.angularVelocity.sqrMagnitude)
    //         );
            
    //     }
    //     if (fell)
    //     {
    //         Done();
    //         AddReward(-1f);
    //     }
    // }

    void OnDrawGizmos()
    {
        if(Application.isPlaying)
        {
            if(visualizeMassDistribution)
            {
                // Gizmos.color = new Color(0,1,1,.5f);
                float drawCOMRadius = 0; //our center of mass radius is relative to the mass of the body part's proportional mass vs the whole body
                totalCharMass = 0;
                foreach(var item in bodyParts)
                {
                    if(item.Value.rb)
                    {
                        totalCharMass += item.Value.rb.mass;
                    }
                }
                foreach(var item in bodyParts)
                {
                    if(item.Value.rb)
                    {
                        Gizmos.color = new Color(0,1,1,.5f);
                        drawCOMRadius = item.Value.rb.mass/totalCharMass;
                        // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(limbRBs[i].transform.up + joints[i].anchor);
                        // var COMPosition = limbRBs[i].transform.TransformPoint(joints[i].anchor);
                        var COMPosition = item.Value.rb.worldCenterOfMass;
                        // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(joints[i].anchor);
                        Gizmos.DrawSphere(COMPosition, drawCOMRadius);

                        // Gizmos.color = Color.red;
                        // // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Value.rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Key.transform.TransformPoint(bodyParts[hips].rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Value.rb.position, drawCOMRadius);
                        // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass, drawCOMRadius);


                    }
                }
                
                
                
                // //limbs
                // for (int i = 0; i < limbs.Length; i++)
                // {
                //     if(limbRBs[i])
                //     {

                //     }
                // }
                // // foreach(Rigidbody rb in limbRBs)
                // // {
                // //     drawCOMRadius = rb.mass/totalCharMass;
                // //     Gizmos.DrawSphere(rb.worldCenterOfMass, drawCOMRadius);
                // // }
                // // body
                // if(bodyRB)
                // {
                //     drawCOMRadius = bodyRB.mass/totalCharMass;
                //     Gizmos.DrawSphere(bodyRB.worldCenterOfMass, drawCOMRadius);
                // }
            }

        }
    }
    // void OnDrawGizmos()
    // {
    //     if(Application.isPlaying)
    //     {
    //         if(visualizeMassDistribution)
    //         {
    //             Gizmos.color = new Color(0,1,1,.5f);
    //             float drawCOMRadius = 0; //our center of mass radius is relative to the mass of the body part's proportional mass vs the whole body
    //             //limbs
    //             for (int i = 0; i < limbs.Length; i++)
    //             {
    //                 if(limbRBs[i])
    //                 {
    //                     drawCOMRadius = limbRBs[i].mass/totalCharMass;
    //                     // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(limbRBs[i].transform.up + joints[i].anchor);
    //                     // var COMPosition = limbRBs[i].transform.TransformPoint(joints[i].anchor);
    //                     var COMPosition = limbRBs[i].worldCenterOfMass;
    //                     // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(joints[i].anchor);
    //                     Gizmos.DrawSphere(COMPosition, drawCOMRadius);

    //                 }
    //             }
    //             // foreach(Rigidbody rb in limbRBs)
    //             // {
    //             //     drawCOMRadius = rb.mass/totalCharMass;
    //             //     Gizmos.DrawSphere(rb.worldCenterOfMass, drawCOMRadius);
    //             // }
    //             // body
    //             if(bodyRB)
    //             {
    //                 drawCOMRadius = bodyRB.mass/totalCharMass;
    //                 Gizmos.DrawSphere(bodyRB.worldCenterOfMass, drawCOMRadius);
    //             }
    //         }

    //     }
    // }

    public override void AgentReset()
    {
        fell = false;
        foreach(var item in bodyParts)
        {
            item.Key.position = item.Value.startingPos;
            item.Key.rotation = item.Value.startingRot;
            item.Value.rb.velocity = Vector3.zero;
            item.Value.rb.angularVelocity = Vector3.zero;
            // item.Value.groundContactScript.touchingGround = false;
        }
        gameObject.transform.rotation = Quaternion.Euler(new Vector3(0, Random.Range(0, 360), 0));





        // Transform[] allChildren = GetComponentsInChildren<Transform>();
        // foreach (Transform child in allChildren)
        // {
        //     if ((child.gameObject.name.Contains("Crawler"))
        //         || (child.gameObject.name.Contains("parent")))
        //     {
        //         continue;
        //     }
        //     child.position = transformsPosition[child.gameObject];
        //     child.rotation = transformsRotation[child.gameObject];
        //     if(child.gameObject.GetComponent<Rigidbody>())
        //     {
        //         child.gameObject.GetComponent<Rigidbody>().velocity = default(Vector3);
        //         child.gameObject.GetComponent<Rigidbody>().angularVelocity = default(Vector3);
        //     }
        // }
        // // gameObject.transform.rotation *= Quaternion.Euler(new Vector3(0, 90, 0));
        // gameObject.transform.rotation = Quaternion.Euler(new Vector3(0, Random.Range(0, 360), 0));
        // // gameObject.transform.rotation = Quaternion.Euler(new Vector3(0, Random.value * 90 - 45, 0));
    }

    public override void AgentOnDone()
    {

    }
}
