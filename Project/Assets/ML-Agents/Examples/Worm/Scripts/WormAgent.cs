using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class WormAgent : Agent
{
    [Range(0, 15)] public float walkingSpeedGoal = 15; //The walking speed to try and achieve
    float m_maxWalkingSpeed = 15; //The max walking speed
    public bool randomizeWalkSpeedEachEpisode; //should the walking speed randomize each episode?

    [Header("Target To Walk Towards")] [Space(10)]
    public TargetController target; //Target the agent will walk towards.

    [Header("Body Parts")] [Space(10)] public Transform bodySegment0;
    public Transform bodySegment1;
    public Transform bodySegment2;
    public Transform bodySegment3;

    [Header("Orientation")] [Space(10)]
    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    public OrientationCubeController orientationCube;

    JointDriveController m_JdController;
    Vector3 m_DirToTarget;
    float m_MovingTowardsDot;
    float m_FacingDot;
    Vector3 m_startingPos;

    [Header("Reward Functions To Use")] [Space(10)]
    public bool rewardMovingTowardsTarget; // Agent should move towards target

    public bool rewardFacingTarget; // Agent should face the target
    public bool rewardUseTimePenalty; // Hurry up

    Quaternion m_LookRotation; //LookRotation from m_TargetDirMatrix to Target
//    Matrix4x4 m_TargetDirMatrix; //Matrix used by agent as orientation reference

    public override void Initialize()
    {
        m_startingPos = bodySegment0.position;
        orientationCube.UpdateOrientation(bodySegment0, target.transform);

        m_JdController = GetComponent<JointDriveController>();
//        m_DirToTarget = target.position - bodySegment0.position;
//        m_LookRotation = Quaternion.LookRotation(m_DirToTarget);
//        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);

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

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        transform.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        orientationCube.UpdateOrientation(bodySegment0, target.transform);
        rewardManager.ResetEpisodeRewards();

        walkingSpeedGoal =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.0f, m_maxWalkingSpeed) : walkingSpeedGoal; //Random Walk Speed
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(orientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(orientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));


        if (bp.rb.transform != bodySegment0)
        {
            //Get position relative to hips in the context of our orientation cube's space
            sensor.AddObservation(
                orientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position));
            sensor.AddObservation(bp.rb.transform.localRotation);
        }
        if(bp.joint)
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        RaycastHit hit;
        float maxDist = 10;
        if (Physics.Raycast(bodySegment0.position, Vector3.down, out hit, maxDist))
        {
            sensor.AddObservation(hit.distance / maxDist);
        }
        else
            sensor.AddObservation(1);

//        sensor.AddObservation(bodySegment0.rotation);
//        sensor.AddObservation(orientationCube.transform.rotation);
        sensor.AddObservation(walkingSpeedGoal);
        sensor.AddObservation(Quaternion.FromToRotation(bodySegment0.forward, orientationCube.transform.forward));

        //Add pos of target relative to orientation cube
        sensor.AddObservation(orientationCube.transform.InverseTransformPoint(target.transform.position));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
        EndEpisode();
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var i = -1;
        // Pick a new target joint rotation
        bpDict[bodySegment0].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[bodySegment1].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[bodySegment2].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
//        bpDict[bodySegment3].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);

        // Update joint strength
        bpDict[bodySegment0].SetJointStrength(vectorAction[++i]);
        bpDict[bodySegment1].SetJointStrength(vectorAction[++i]);
        bpDict[bodySegment2].SetJointStrength(vectorAction[++i]);
//        bpDict[bodySegment3].SetJointStrength(vectorAction[++i]);

        //Reset if Worm fell through floor;
        if (bodySegment0.position.y < m_startingPos.y - 2)
        {
            EndEpisode();
        }
    }
//
//    public override void OnActionReceived(float[] vectorAction)
//    {
//        // The dictionary with all the body parts in it are in the jdController
//        var bpDict = m_JdController.bodyPartsDict;
//
//        var i = -1;
//        // Pick a new target joint rotation
//        bpDict[bodySegment1].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
//        bpDict[bodySegment2].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
//        bpDict[bodySegment3].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
//
//        // Update joint strength
//        bpDict[bodySegment1].SetJointStrength(vectorAction[++i]);
//        bpDict[bodySegment2].SetJointStrength(vectorAction[++i]);
//        bpDict[bodySegment3].SetJointStrength(vectorAction[++i]);
//
//        //Reset if Worm fell through floor;
//        if (bodySegment0.position.y < m_startingPos.y - 2)
//        {
//            EndEpisode();
//        }
//    }

    void FixedUpdate()
    {
        orientationCube.UpdateOrientation(bodySegment0, target.transform);

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
        
        velReward =
            Mathf.Exp(-0.1f * (orientationCube.transform.forward * walkingSpeedGoal -
                               m_JdController.bodyPartsDict[bodySegment0].rb.velocity).sqrMagnitude);
        facingReward = 0.5f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward) +
                       0.5f * Vector3.Dot(orientationCube.transform.up, bodySegment0.up);
        rewardManager.UpdateReward("velFacingComboReward", velReward * facingReward);

    }

    public RewardManager rewardManager;
    public float velReward;

    /// <summary>
    /// Reward moving towards target & Penalize moving away from target.
    /// </summary>
    void RewardFunctionMovingTowards()
    {
//        velReward = Vector3.Dot(orientationCube.transform.forward,
//            Vector3.ClampMagnitude(m_JdController.bodyPartsDict[bodySegment0].rb.velocity, maximumWalkingSpeed));
        velReward =
            Mathf.Exp(-0.1f * (orientationCube.transform.forward * walkingSpeedGoal -
                               m_JdController.bodyPartsDict[bodySegment0].rb.velocity).sqrMagnitude);
//        velReward = Vector3.Dot(orientationCube.transform.forward,
//            m_JdController.bodyPartsDict[bodySegment0].rb.velocity);
//        rewardManager.UpdateReward("velReward", velReward);
//        rewardManager.UpdateReward("velReward", (velReward/maximumWalkingSpeed)/MaxStep);
//        rewardManager.UpdateReward("velReward", (velReward / maximumWalkingSpeed));
        rewardManager.UpdateReward("velReward", velReward);


//        m_MovingTowardsDot = Vector3.Dot(orientationCube.transform.forward, m_JdController.bodyPartsDict[bodySegment0].rb.velocity);
//        AddReward(0.01f * m_MovingTowardsDot);
    }

    public float facingReward;

    /// <summary>
    /// Reward facing target & Penalize facing away from target
    /// </summary>
    void RewardFunctionFacingTarget()
    {
//        facingReward =  Quaternion.Dot(orientationCube.transform.rotation, bodySegment0.rotation);
//        facingReward = Quaternion.Dot(bodySegment0.rotation, orientationCube.transform.rotation);
//        print(Vector3.Dot(bodySegment0.forward, orientationCube.transform.forward));
//        rewardManager.UpdateReward("facingReward", facingReward);

//        float bodyRotRelativeToMatrixDot = Quaternion.Dot(orientationCube.transform.rotation, bodySegment0.rotation);
//        AddReward(0.01f * bodyRotRelativeToMatrixDot);

        //normalizes between (-1, 1)
        facingReward = 0.5f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward) +
            0.5f * Vector3.Dot(orientationCube.transform.up, bodySegment0.up);
//        facingReward =  Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward);
//        rewardManager.UpdateReward("facingReward", facingReward);
        rewardManager.UpdateReward("facingReward", facingReward);
//        rewardManager.UpdateReward("facingReward", facingReward/MaxStep);
//        AddReward(0.01f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward));
    }

    /// <summary>
    /// Existential penalty for time-contrained tasks.
    /// </summary>
    void RewardFunctionTimePenalty()
    {
        AddReward(-0.001f);
    }
}