using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerAgent : Agent
{
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
    public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();

    [System.Serializable]
    public class BodyPart
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public Vector3 startingPos;
        public Quaternion startingRot;
        public WalkerGroundContact groundContactScript;
    }

    public void SetupBodyPart(Transform t)
    {
        BodyPart bp = new BodyPart
        {
            rb = t.GetComponent<Rigidbody>(),
            joint = t.GetComponent<ConfigurableJoint>(),
            startingPos = t.position,
            startingRot = t.rotation
        };
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

    }

    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return (Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    public void BodyPartObservation(BodyPart bp)
    {
        var rb = bp.rb;
        AddVectorObs(bp.groundContactScript.touchingGround ? 1 : 0); // Is this bp touching the ground
        bp.groundContactScript.touchingGround = false;

        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToHips);

        if (bp.joint && (bp.rb.transform != handL && bp.rb.transform != handR))
        {
            var jointRotation = GetJointRotation(bp.joint);
            AddVectorObs(jointRotation); // Get the joint rotation
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
        x = (x + 1f) * 0.5f;
        y = (y + 1f) * 0.5f;
        z = (z + 1f) * 0.5f;

        var xRot = Mathf.Lerp(bp.joint.lowAngularXLimit.limit, bp.joint.highAngularXLimit.limit, x);
        var yRot = Mathf.Lerp(-bp.joint.angularYLimit.limit, bp.joint.angularYLimit.limit, y);
        var zRot = Mathf.Lerp(-bp.joint.angularZLimit.limit, bp.joint.angularZLimit.limit, z);

        bp.joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
        var jd = new JointDrive
        {
            positionSpring = ((strength + 1f) * 0.5f) * 10000f,
            maximumForce = 250000f
        };
        bp.joint.slerpDrive = jd;
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
            + 0.01f * Vector3.Dot(goalDirection, bodyParts[chest].rb.transform.forward)
            + 0.01f * bodyParts[head].rb.position.y
            + 0.03f * Vector3.Dot(goalDirection, bodyParts[chest].rb.velocity)
        );
    }

    public override void AgentReset()
    {
        foreach (var item in bodyParts)
        {
            item.Key.position = item.Value.startingPos;
            item.Key.rotation = item.Value.startingRot;
            item.Value.rb.velocity = Vector3.zero;
            item.Value.rb.angularVelocity = Vector3.zero;
        }
    }

}