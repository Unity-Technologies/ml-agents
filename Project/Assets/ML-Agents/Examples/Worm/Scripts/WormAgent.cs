using UnityEngine;
using Unity.MLAgents;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class WormAgent : Agent
{
    [Range(0, 15)]
//    private float m_TargetWalkingSpeed = 10;
    const float m_maxWalkingSpeed = 15; //The max walking speed


    //Brains
    //A different brain will be used depending on the CrawlerAgentBehaviorType selected
    [Header("NN Models")]
    public NNModel wormDyBrain;
    public NNModel wormStBrain;

//    public float walkingSpeedGoal = 15; //The walking speed to try and achieve
//    float m_maxWalkingSpeed = 15; //The max walking speed
//    public bool randomizeWalkSpeedEachEpisode; //should the walking speed randomize each episode?

    [Header("Target Prefabs")]
    public Transform dynamicTargetPrefab;
    public Transform staticTargetPrefab;
    private Transform m_Target; //Target the agent will walk towards.

    [Header("Body Parts")] [Space(10)] public Transform bodySegment0;
    public Transform bodySegment1;
    public Transform bodySegment2;
    public Transform bodySegment3;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
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

    public enum WormAgentBehaviorType
    {
        WormDynamic, WormStatic
    }

    public WormAgentBehaviorType typeOfWorm;
    public override void Initialize()
    {
        var m_BehaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        switch (typeOfWorm)
        {
            case WormAgentBehaviorType.WormDynamic:
            {
                m_BehaviorParams.BehaviorName = "CrawlerDynamic";
                if (wormDyBrain)
                    m_BehaviorParams.Model = wormDyBrain;
                m_Target = Instantiate(dynamicTargetPrefab, transform.position, Quaternion.identity, transform);
                break;
            }
            case WormAgentBehaviorType.WormStatic:
            {
                m_BehaviorParams.BehaviorName = "CrawlerDynamicVariableSpeed";
                if (wormStBrain)
                    m_BehaviorParams.Model = wormStBrain;
                m_Target = Instantiate(staticTargetPrefab, transform.position, Quaternion.identity, transform);
                break;
            }
        }

        m_startingPos = bodySegment0.position;
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();

        m_OrientationCube.UpdateOrientation(bodySegment0, m_Target.transform);

        m_JdController = GetComponent<JointDriveController>();

        //Setup each body part
        m_JdController.SetupBodyPart(bodySegment0);
        m_JdController.SetupBodyPart(bodySegment1);
        m_JdController.SetupBodyPart(bodySegment2);
        m_JdController.SetupBodyPart(bodySegment3);

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
        bodySegment0.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

//        m_OrientationCube.UpdateOrientation(bodySegment0, target.transform);
        rewardManager.ResetEpisodeRewards();

//        walkingSpeedGoal =
//            randomizeWalkSpeedEachEpisode ? Random.Range(0.0f, m_maxWalkingSpeed) : walkingSpeedGoal; //Random Walk Speed
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
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));


        if (bp.rb.transform != bodySegment0)
        {
            //Get position relative to hips in the context of our orientation cube's space
            sensor.AddObservation(
                m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position));
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

        var cubeForward = m_OrientationCube.transform.forward;

//        sensor.AddObservation(bodySegment0.rotation);
//        sensor.AddObservation(orientationCube.transform.rotation);
        //velocity we want to match
        var velGoal = cubeForward * m_maxWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));




//        sensor.AddObservation(walkingSpeedGoal);
        sensor.AddObservation(Quaternion.FromToRotation(bodySegment0.forward, cubeForward));

        //Add pos of target relative to orientation cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

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
//        EndEpisode();
    }

    //Returns the average velocity of all of the body parts
    //Using the velocity of the hips only has shown to result in more erratic movement from the limbs, so...
    //...using the average helps prevent this erratic movement
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        Vector3 avgVel = Vector3.zero;

        //ALL RBS
        int numOfRB = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRB++;
            velSum += item.rb.velocity;
        }

        avgVel = velSum / numOfRB;
        return avgVel;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var i = -1;
        var continuousActions = actionBuffers.ContinuousActions;
        // Pick a new target joint rotation
        bpDict[bodySegment0].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[bodySegment1].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[bodySegment2].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        // Update joint strength
        bpDict[bodySegment0].SetJointStrength(continuousActions[++i]);
        bpDict[bodySegment1].SetJointStrength(continuousActions[++i]);
        bpDict[bodySegment2].SetJointStrength(continuousActions[++i]);

        //Reset if Worm fell through floor;
        if (bodySegment0.position.y < m_startingPos.y - 2)
        {
            EndEpisode();
        }
    }

    void FixedUpdate()
    {
//        m_OrientationCube.UpdateOrientation(bodySegment0, target.transform);
        UpdateOrientationObjects();
//        // Set reward for this step according to mixture of the following elements.
//        if (rewardMovingTowardsTarget)
//        {
//            RewardFunctionMovingTowards();
//        }
//
//        if (rewardFacingTarget)
//        {
//            RewardFunctionFacingTarget();
//        }
//
//        if (rewardUseTimePenalty)
//        {
//            RewardFunctionTimePenalty();
//        }

        var velReward =
            Mathf.Exp(-0.1f * (m_OrientationCube.transform.forward * m_maxWalkingSpeed -
                               m_JdController.bodyPartsDict[bodySegment0].rb.velocity).sqrMagnitude);
        var lookDotForward = (Vector3.Dot(m_OrientationCube.transform.forward, bodySegment0.forward) + 1) * .5F;
        var lookDotUp = (Vector3.Dot(m_OrientationCube.transform.up, bodySegment0.up) + 1) * .5F;
        rewardManager.UpdateReward("velFacingComboReward", velReward * (lookDotForward + lookDotUp));


//        facingReward = 0.5f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward) +
//                       0.5f * Vector3.Dot(orientationCube.transform.up, bodySegment0.up);
//        rewardManager.UpdateReward("velFacingComboReward", velReward * facingReward);

    }

    //Update OrientationCube and DirectionIndicator
    void UpdateOrientationObjects()
    {
//        m_WorldDirToWalk = target.position - hips.position;
        m_OrientationCube.UpdateOrientation(bodySegment0, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    public RewardManager rewardManager;
//    public float velReward;

//    /// <summary>
//    /// Reward moving towards target & Penalize moving away from target.
//    /// </summary>
//    void RewardFunctionMovingTowards()
//    {
////        velReward = Vector3.Dot(orientationCube.transform.forward,
////            Vector3.ClampMagnitude(m_JdController.bodyPartsDict[bodySegment0].rb.velocity, maximumWalkingSpeed));
//        velReward =
//            Mathf.Exp(-0.1f * (orientationCube.transform.forward * walkingSpeedGoal -
//                               m_JdController.bodyPartsDict[bodySegment0].rb.velocity).sqrMagnitude);
////        velReward = Vector3.Dot(orientationCube.transform.forward,
////            m_JdController.bodyPartsDict[bodySegment0].rb.velocity);
////        rewardManager.UpdateReward("velReward", velReward);
////        rewardManager.UpdateReward("velReward", (velReward/maximumWalkingSpeed)/MaxStep);
////        rewardManager.UpdateReward("velReward", (velReward / maximumWalkingSpeed));
//        rewardManager.UpdateReward("velReward", velReward);
//
//
////        m_MovingTowardsDot = Vector3.Dot(orientationCube.transform.forward, m_JdController.bodyPartsDict[bodySegment0].rb.velocity);
////        AddReward(0.01f * m_MovingTowardsDot);
//    }
//
////    public float facingReward;
//
//    /// <summary>
//    /// Reward facing target & Penalize facing away from target
//    /// </summary>
//    void RewardFunctionFacingTarget()
//    {
////        facingReward =  Quaternion.Dot(orientationCube.transform.rotation, bodySegment0.rotation);
////        facingReward = Quaternion.Dot(bodySegment0.rotation, orientationCube.transform.rotation);
////        print(Vector3.Dot(bodySegment0.forward, orientationCube.transform.forward));
////        rewardManager.UpdateReward("facingReward", facingReward);
//
////        float bodyRotRelativeToMatrixDot = Quaternion.Dot(orientationCube.transform.rotation, bodySegment0.rotation);
////        AddReward(0.01f * bodyRotRelativeToMatrixDot);
//        // b. Rotation alignment with target direction.
//
//        //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
//        var lookDirForward = (Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward) + 1) * .5F;
//        var lookDirUp = (Vector3.Dot(orientationCube.transform.up, bodySegment0.up) + 1) * .5F;
//
//        facingReward = lookDirForward + lookDirUp;
//        rewardManager.UpdateReward("facingReward", facingReward);
//
////        //normalizes between (-1, 1)
////        facingReward = 0.5f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward) +
////            0.5f * Vector3.Dot(orientationCube.transform.up, bodySegment0.up);
//////        facingReward =  Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward);
//////        rewardManager.UpdateReward("facingReward", facingReward);
////        rewardManager.UpdateReward("facingReward", facingReward);
////        rewardManager.UpdateReward("facingReward", facingReward/MaxStep);
////        AddReward(0.01f * Vector3.Dot(orientationCube.transform.forward, bodySegment0.forward));
//    }
//
//    /// <summary>
//    /// Existential penalty for time-contrained tasks.
//    /// </summary>
//    void RewardFunctionTimePenalty()
//    {
//        AddReward(-0.001f);
//    }
}
