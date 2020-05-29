using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class CrawlerAgent : Agent
{
    [Header("Target To Walk Towards")]
    [Space(10)]
    public Transform target;
    Vector3 m_WalkDir; //Direction to the target
    Quaternion m_WalkDirLookRot; //Will hold the rotation to our target

    [Space(10)]
    [Header("Orientation Cube")]
    [Space(10)]
    //This will be used as a stable observation platform for the ragdoll to use.
    GameObject m_OrientationCube;

    public Transform ground;
    public bool detectTargets;
    public bool targetIsStatic;
    public bool respawnTargetWhenTouched;
    public float targetSpawnRadius;

    [Header("Body Parts")] [Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    [Header("Joint Settings")] [Space(10)] JointDriveController m_JdController;
    float m_MovingTowardsDot;
    float m_FacingDot;

    [Header("Reward Functions To Use")]
    [Space(10)]
    public bool rewardMovingTowardsTarget; // Agent should move towards target

    public bool rewardFacingTarget; // Agent should face the target
    public bool rewardUseTimePenalty; // Hurry up

    [Header("Foot Grounded Visualization")]
    [Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    Quaternion m_LookRotation;
    Matrix4x4 m_TargetDirMatrix;

    public override void Initialize()
    {
        //Spawn an orientation cube
        Vector3 oCubePos = body.position;
        oCubePos.y = -.45f;
        m_OrientationCube = Instantiate(Resources.Load<GameObject>("OrientationCube"), oCubePos, Quaternion.identity);
        m_OrientationCube.transform.SetParent(transform);
        UpdateOrientationCube();

        m_JdController = GetComponent<JointDriveController>();

        //Setup each body part
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);
    }
    
    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        var rb = bp.rb;
        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));
        
        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - body.position));

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

//    /// <summary>
//    /// Add relevant information on each body part to observations.
//    /// </summary>
//    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
//    {
//        var rb = bp.rb;
//        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground
//
//        var velocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(rb.velocity);
//        sensor.AddObservation(velocityRelativeToLookRotationToTarget);
//
//        var angularVelocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(rb.angularVelocity);
//        sensor.AddObservation(angularVelocityRelativeToLookRotationToTarget);
//
//        if (bp.rb.transform != body)
//        {
//            var localPosRelToBody = body.InverseTransformPoint(rb.position);
//            sensor.AddObservation(localPosRelToBody);
//            sensor.AddObservation(bp.currentXNormalizedRot); // Current x rot
//            sensor.AddObservation(bp.currentYNormalizedRot); // Current y rot
//            sensor.AddObservation(bp.currentZNormalizedRot); // Current z rot
//            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
//        }
//    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, m_OrientationCube.transform.forward));
        
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(target.position));
       
        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
        {
            sensor.AddObservation(hit.distance/maxRaycastDist);
        }
        else
            sensor.AddObservation(maxRaycastDist);
        
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }
    
//    public override void CollectObservations(VectorSensor sensor)
//    {
//        m_JdController.GetCurrentJointForces();
//
//        // Update pos to target
//        m_DirToTarget = target.position - body.position;
//        m_LookRotation = Quaternion.LookRotation(m_DirToTarget);
//        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);
//
//        RaycastHit hit;
//        if (Physics.Raycast(body.position, Vector3.down, out hit, 10.0f))
//        {
//            sensor.AddObservation(hit.distance);
//        }
//        else
//            sensor.AddObservation(10.0f);
//
//        // Forward & up to help with orientation
//        var bodyForwardRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(body.forward);
//        sensor.AddObservation(bodyForwardRelativeToLookRotationToTarget);
//
//        var bodyUpRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(body.up);
//        sensor.AddObservation(bodyUpRelativeToLookRotationToTarget);
//
//        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
//        {
//            CollectObservationBodyPart(bodyPart, sensor);
//        }
//    }

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
        bpDict[leg0Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(vectorAction[++i], 0, 0);

        // Update joint strength
        bpDict[leg0Upper].SetJointStrength(vectorAction[++i]);
        bpDict[leg1Upper].SetJointStrength(vectorAction[++i]);
        bpDict[leg2Upper].SetJointStrength(vectorAction[++i]);
        bpDict[leg3Upper].SetJointStrength(vectorAction[++i]);
        bpDict[leg0Lower].SetJointStrength(vectorAction[++i]);
        bpDict[leg1Lower].SetJointStrength(vectorAction[++i]);
        bpDict[leg2Lower].SetJointStrength(vectorAction[++i]);
        bpDict[leg3Lower].SetJointStrength(vectorAction[++i]);
    }
    void UpdateOrientationCube()
    {
        //FACING DIR
        m_WalkDir = target.position - m_OrientationCube.transform.position;
        m_WalkDir.y = 0; //flatten dir on the y
        m_WalkDirLookRot = Quaternion.LookRotation(m_WalkDir); //get our look rot to the target
        
        //UPDATE ORIENTATION CUBE POS & ROT
        m_OrientationCube.transform.position = body.position;
        m_OrientationCube.transform.rotation = m_WalkDirLookRot;
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
            
            UpdateOrientationCube();
        }

        // If enabled the feet will light up green when the foot is grounded.
        // This is just a visualization and isn't necessary for function
        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
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
        m_MovingTowardsDot = Vector3.Dot(m_OrientationCube.transform.forward, m_JdController.bodyPartsDict[body].rb.velocity);
        AddReward(0.03f * m_MovingTowardsDot);
    }

    /// <summary>
    /// Reward facing target & Penalize facing away from target
    /// </summary>
    void RewardFunctionFacingTarget()
    {
        AddReward(0.01f * Vector3.Dot(m_OrientationCube.transform.forward, body.forward));
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
        
        //Random start rotation to help generalize
        transform.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationCube();

        if (detectTargets && !targetIsStatic)
        {
            GetRandomTargetPos();
        }
    }
    
    private void OnDrawGizmosSelected()
    {
        if (Application.isPlaying)
        {   
            Gizmos.color = Color.green;
            Gizmos.matrix = m_OrientationCube.transform.localToWorldMatrix;
            Gizmos.DrawWireCube(Vector3.zero, m_OrientationCube.transform.localScale);
            Gizmos.DrawRay(Vector3.zero, Vector3.forward);
        }
    }
}
