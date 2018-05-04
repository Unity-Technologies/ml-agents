using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerAgent : Agent
{
    public bool fell;
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
    public float targetJointAngularVelocityScalar;
    public Vector3 forwardDir;
    public float maxBodyPartVelocity = 3;
    public float torquePenalty;
    public float velocityPenalty;
    public float torquePenaltyFinal;
    public float velocityPenaltyFinal;
    public float chestHeightRewardFinal;
    public float chestYUpRewardFinal;
    public float facingDirReward;
    [SerializeField] private float[] actionValues;

    [System.Serializable]
    public class BodyPart
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public Vector3 startingPos;
        public Quaternion startingRot;
        public float currentEnergyLevel;
        public WalkerGroundContact groundContactScript;
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

        totalCharMass = 0; //reset to 0

        foreach (var item in bodyParts)
        {
            if (item.Value.rb)
            {
                item.Value.rb.maxAngularVelocity = 500;
                totalCharMass += item.Value.rb.mass;
            }
        }
    }

    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return (Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    public void BodyPartObservation(BodyPart bp)
    {
        var rb = bp.rb;
        AddVectorObs(bp.groundContactScript.touchingGround ? 1 : 0); //is this bp touching the ground
        bp.groundContactScript.touchingGround = false; //reset

        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToHips);

        if (bp.joint)
        {
            if (bp.rb.transform != handL && bp.rb.transform != handR)
            {
                var jointRotation = GetJointRotation(bp.joint);
                AddVectorObs(jointRotation); //get the joint rotation
            }
        }
    }

    public override void CollectObservations()
    {
        foreach (var item in bodyParts)
        {
            BodyPartObservation(item.Value);
        }
    }

    public void SetNormalizedTargetRotation(BodyPart bp, float x, float y, float z, float strength)
    {
        float xRot = 0;
        float yRot = 0;
        float zRot = 0;

        x = (x + 1f) * 0.5f;
        xRot = Mathf.Lerp(bp.joint.lowAngularXLimit.limit, bp.joint.highAngularXLimit.limit, x);
        y = (y + 1f) * 0.5f;
        yRot = Mathf.Lerp(-bp.joint.angularYLimit.limit, bp.joint.angularYLimit.limit, y);
        z = (z + 1f) * 0.5f;
        zRot = Mathf.Lerp(-bp.joint.angularZLimit.limit, bp.joint.angularZLimit.limit, z);

        bp.joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
        JointDrive jd = new JointDrive
        {
            positionSpring = ((strength + 1f) * 0.5f) * 10000f,
            maximumForce = 250000f
        };
        bp.joint.slerpDrive = jd;
    }

    //the dot product of two vectors facing exactly the same dir is 1. 
    //if two vectors are facing exactly opposite dir, the dp is -1.
    //a dp of say .9 means two vectors are almost pointing the same way.
    //a dp of 0 means two vectors are perpendicular (i.e. 90 degree diff)
    //we can use DotProduct to determine if the ragdoll is facing our target dir or not
    float FacingTargetDirDot(Vector3 currentDir, Vector3 targetDir)
    {
        facingDirReward = Vector3.Dot(currentDir, targetDir);
        return facingDirReward;
    }

    bool IsFacingTargetDir()
    {
        bool facingDir = false;
        var targetDir = Vector3.right;
        Vector3 forwardDir = bodyParts[hips].rb.transform.forward;
        forwardDir.y = 0;
        forwardDir.z = 0;
        float facingTowardsDot = Vector3.Dot(forwardDir, targetDir);
        if (facingTowardsDot >= .9f) //roughly facing dir
        {
            facingDir = true;
        }

        return facingDir;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {

        SetNormalizedTargetRotation(bodyParts[chest], vectorAction[0], vectorAction[1], vectorAction[2],
            vectorAction[26]);
        SetNormalizedTargetRotation(bodyParts[spine], vectorAction[3], vectorAction[4], vectorAction[5],
            vectorAction[27]);

        SetNormalizedTargetRotation(bodyParts[thighL], vectorAction[6], vectorAction[7], 0, vectorAction[28]);
        SetNormalizedTargetRotation(bodyParts[shinL], vectorAction[8], 0, 0, vectorAction[29]);
        SetNormalizedTargetRotation(bodyParts[footL], vectorAction[9], vectorAction[10], vectorAction[11],
            vectorAction[30]);
        SetNormalizedTargetRotation(bodyParts[thighR], vectorAction[12], vectorAction[13], 0, vectorAction[31]);
        SetNormalizedTargetRotation(bodyParts[shinR], vectorAction[14], 0, 0, vectorAction[32]);
        SetNormalizedTargetRotation(bodyParts[footR], vectorAction[15], vectorAction[16], vectorAction[17],
            vectorAction[33]);

        SetNormalizedTargetRotation(bodyParts[armL], vectorAction[18], vectorAction[19], 0, vectorAction[34]);
        SetNormalizedTargetRotation(bodyParts[forearmL], vectorAction[20], 0, 0, vectorAction[34]);
        SetNormalizedTargetRotation(bodyParts[armR], vectorAction[21], vectorAction[22], 0, vectorAction[36]);
        SetNormalizedTargetRotation(bodyParts[forearmR], vectorAction[23], 0, 0, vectorAction[37]);
        SetNormalizedTargetRotation(bodyParts[head], vectorAction[24], vectorAction[25], 0, vectorAction[38]);

        AddReward(
            - 0.01f * Mathf.Abs(bodyParts[chest].rb.position.z - transform.position.z)
            - 0.01f * Mathf.Abs(bodyParts[head].rb.velocity.y)
            - 0.01f * Mathf.Abs(bodyParts[head].rb.velocity.z)
            + 0.01f * Vector3.Dot(Vector3.right, bodyParts[chest].rb.transform.forward)
            + 0.01f * bodyParts[head].rb.position.y
            + 0.03f * bodyParts[chest].rb.velocity.x
        );
    }

    void OnDrawGizmos()
    {
        if (Application.isPlaying)
        {
            if (visualizeMassDistribution)
            {
                totalCharMass = 0;
                foreach (var item in bodyParts)
                {
                    if (item.Value.rb)
                    {
                        totalCharMass += item.Value.rb.mass;
                    }
                }

                foreach (var item in bodyParts)
                {
                    if (item.Value.rb)
                    {
                        Gizmos.color = new Color(0, 1, 1, .5f);
                        var drawComRadius = item.Value.rb.mass / totalCharMass;
                        var comPosition = item.Value.rb.worldCenterOfMass;
                        Gizmos.DrawSphere(comPosition, drawComRadius);
                    }
                }
            }
        }
    }

    public override void AgentReset()
    {
        fell = false;
        foreach (var item in bodyParts)
        {
            item.Key.position = item.Value.startingPos;
            item.Key.rotation = item.Value.startingRot;
            item.Value.rb.velocity = Vector3.zero;
            item.Value.rb.angularVelocity = Vector3.zero;
        }
    }

    public override void AgentOnDone()
    {
    }
}