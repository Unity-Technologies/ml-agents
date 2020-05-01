using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class WormAgent : Agent
{
    [Header("Target To Walk Towards")]
    [Space(10)]
    public Transform target;

    public Transform ground;
    public bool detectTargets;
    public bool targetIsStatic;
    public bool respawnTargetWhenTouched;
    public float targetSpawnRadius;

    [Header("Body Parts")] [Space(10)]
    public Transform bodySegment0;
    public Transform bodySegment1;
    public Transform bodySegment2;
    public Transform bodySegment3;

    [Header("Joint Settings")] [Space(10)]
    JointDriveController m_JdController;
    Vector3 m_DirToTarget;
    float m_MovingTowardsDot;
    float m_FacingDot;

    [Header("Reward Functions To Use")]
    [Space(10)]
    public bool rewardMovingTowardsTarget; // Agent should move towards target

    public bool rewardFacingTarget; // Agent should face the target
    public bool rewardUseTimePenalty; // Hurry up

    Quaternion m_LookRotation; //LookRotation from m_TargetDirMatrix to Target
    Matrix4x4 m_TargetDirMatrix; //Matrix used by agent as orientation reference

    public override void Initialize()
    {
        m_JdController = GetComponent<JointDriveController>();
        m_DirToTarget = target.position - bodySegment0.position;
        m_LookRotation = Quaternion.LookRotation(m_DirToTarget);
        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);

        //Setup each body part
        m_JdController.SetupBodyPart(bodySegment0);
        m_JdController.SetupBodyPart(bodySegment1);
        m_JdController.SetupBodyPart(bodySegment2);
        m_JdController.SetupBodyPart(bodySegment3);

        //We only want the head to detect the target
        //So we need to remove TargetContact from everything else
        //This is a temp fix till we can redesign
        DestroyImmediate(bodySegment1.GetComponent<TargetContact>());
        DestroyImmediate(bodySegment2.GetComponent<TargetContact>());
        DestroyImmediate(bodySegment3.GetComponent<TargetContact>());
    }


    //Get Joint Rotation Relative to the Connected Rigidbody
    //We want to collect this info because it is the actual rotation, not the "target rotation"
    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return(Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        var rb = bp.rb;
        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground

        var velocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(rb.velocity);
        sensor.AddObservation(velocityRelativeToLookRotationToTarget);

        var angularVelocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(rb.angularVelocity);
        sensor.AddObservation(angularVelocityRelativeToLookRotationToTarget);

        if (bp.rb.transform != bodySegment0)
        {
            var localPosRelToBody = bodySegment0.InverseTransformPoint(rb.position);
            sensor.AddObservation(localPosRelToBody);
            sensor.AddObservation(GetJointRotation(bp.joint));
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        m_JdController.GetCurrentJointForces();

        // Update pos to target
        m_DirToTarget = target.position - bodySegment0.position;
        m_LookRotation = Quaternion.LookRotation(m_DirToTarget);
        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);

        RaycastHit hit;
        float maxDist = 10;
        if (Physics.Raycast(bodySegment0.position, Vector3.down, out hit, maxDist))
        {
            sensor.AddObservation(hit.distance/maxDist);
        }
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }

        //Rotation delta between the matrix and the head
        Quaternion headRotationDeltaFromMatrixRot = Quaternion.Inverse(m_TargetDirMatrix.rotation) * bodySegment0.rotation;
        sensor.AddObservation(headRotationDeltaFromMatrixRot);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
        if (respawnTargetWhenTouched)
        {
            GetRandomTargetPos();
        }
    }

    /// <summary>
    /// Moves target to a random position within specified radius.
    /// </summary>
    public void GetRandomTargetPos()
    {
        var newTargetPos = Random.insideUnitSphere * targetSpawnRadius;
        newTargetPos.y = 5;
        target.position = newTargetPos + ground.position;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var i = -1;
        // Pick a new target joint rotation
        bpDict[bodySegment1].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[bodySegment2].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[bodySegment3].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);

        // Update joint strength
        bpDict[bodySegment1].SetJointStrength(vectorAction[++i]);
        bpDict[bodySegment2].SetJointStrength(vectorAction[++i]);
        bpDict[bodySegment3].SetJointStrength(vectorAction[++i]);

        if (bodySegment0.position.y < ground.position.y -2)
        {
            EndEpisode();
        }
    }

    void FixedUpdate()
    {
        if (detectTargets)
        {
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                if (bodyPart.targetContact && bodyPart.targetContact.touchingTarget)
                {
                    TouchedTarget();
                }
            }
        }

        // Set reward for this step according to mixture of the following elements.
        if (rewardMovingTowardsTarget)
        {
            RewardFunctionMovingTowards();
        }

        if (rewardFacingTarget)
        {
            RewardFunctionFacingTarget();
        }

        if (rewardUseTimePenalty)
        {
            RewardFunctionTimePenalty();
        }
    }

    /// <summary>
    /// Reward moving towards target & Penalize moving away from target.
    /// </summary>
    void RewardFunctionMovingTowards()
    {
        m_MovingTowardsDot = Vector3.Dot(m_JdController.bodyPartsDict[bodySegment0].rb.velocity, m_DirToTarget.normalized);
        AddReward(0.01f * m_MovingTowardsDot);
    }

    /// <summary>
    /// Reward facing target & Penalize facing away from target
    /// </summary>
    void RewardFunctionFacingTarget()
    {
        float bodyRotRelativeToMatrixDot = Quaternion.Dot(m_TargetDirMatrix.rotation, bodySegment0.rotation);
        AddReward(0.01f * bodyRotRelativeToMatrixDot);
    }

    /// <summary>
    /// Existential penalty for time-contrained tasks.
    /// </summary>
    void RewardFunctionTimePenalty()
    {
        AddReward(-0.001f);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
        if (m_DirToTarget != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(m_DirToTarget);
        }
        transform.Rotate(Vector3.up, Random.Range(0.0f, 360.0f));

        if (!targetIsStatic)
        {
            GetRandomTargetPos();
        }
    }
}
