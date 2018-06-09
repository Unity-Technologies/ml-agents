using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

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

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset()
        {
            rb.transform.position = startingPos;
            rb.transform.rotation = startingRot;
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
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
                positionSpring = ((strength + 1f) * 0.5f) * 10000f,
                maximumForce = 250000f
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
        bodyParts.Add(t, bp);
        bp.groundContact = t.GetComponent<GroundContact>();
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
        bp.groundContact.touchingGround = false;

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

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations()
    {
        AddVectorObs(goalDirection);
        foreach (var bodyPart in bodyParts.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        // Apply action to all relevant body parts. 
        
        bodyParts[chest].SetNormalizedTargetRotation(vectorAction[0], vectorAction[1], vectorAction[2],
            vectorAction[26]);
        bodyParts[spine].SetNormalizedTargetRotation(vectorAction[3], vectorAction[4], vectorAction[5],
            vectorAction[27]);

        bodyParts[thighL].SetNormalizedTargetRotation(vectorAction[6], vectorAction[7], 0, vectorAction[28]);
        bodyParts[shinL].SetNormalizedTargetRotation(vectorAction[8], 0, 0, vectorAction[29]);
        bodyParts[footL].SetNormalizedTargetRotation(vectorAction[9], vectorAction[10], vectorAction[11],
            vectorAction[30]);
        
        bodyParts[thighR].SetNormalizedTargetRotation(vectorAction[12], vectorAction[13], 0, vectorAction[31]);
        bodyParts[shinR].SetNormalizedTargetRotation(vectorAction[14], 0, 0, vectorAction[32]);
        bodyParts[footR].SetNormalizedTargetRotation(vectorAction[15], vectorAction[16], vectorAction[17],
            vectorAction[33]);

        bodyParts[armL].SetNormalizedTargetRotation(vectorAction[18], vectorAction[19], 0, vectorAction[34]);
        bodyParts[forearmL].SetNormalizedTargetRotation(vectorAction[20], 0, 0, vectorAction[34]);
        
        bodyParts[armR].SetNormalizedTargetRotation(vectorAction[21], vectorAction[22], 0, vectorAction[36]);
        bodyParts[forearmR].SetNormalizedTargetRotation(vectorAction[23], 0, 0, vectorAction[37]);
        
        bodyParts[head].SetNormalizedTargetRotation(vectorAction[24], vectorAction[25], 0, vectorAction[38]);

        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        // d. Discourage head movement.
        AddReward(
            + 0.03f * Vector3.Dot(goalDirection, bodyParts[hips].rb.velocity)
            + 0.01f * Vector3.Dot(goalDirection, hips.forward)
            + 0.01f * (head.position.y - hips.position.y)
            - 0.01f * Vector3.Distance(bodyParts[head].rb.velocity, bodyParts[hips].rb.velocity)
        );
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
