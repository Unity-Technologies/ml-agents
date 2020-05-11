using MLAgentsExamples;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

public class WalkerAgentDynamic : Agent
{
    [Header("Specific to Walker")]
    [Header("Target To Walk Towards")]
    [Space(10)]
    public Transform target;
    public Transform ground;
    public bool detectTargets;
    public bool targetIsStatic;
    public bool respawnTargetWhenTouched;
    public float targetSpawnRadius;
    [Header("Walk Direction Worldspace")] 
//    public Vector3 walkDirWorldspace = Vector3.right;

    
    //ORIENTATION
    Vector3 m_WalkDir;
    Quaternion m_WalkDirLookRot;
    Matrix4x4 m_worldPosMatrix;
    
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
    JointDriveController m_JdController;

    Rigidbody m_HipsRb;
    Rigidbody m_ChestRb;
    Rigidbody m_SpineRb;

    EnvironmentParameters m_ResetParams;

    private GameObject m_OrientationCube;
    public override void Initialize()
    {
        Vector3 oCubePos = hips.position;
        oCubePos.y = -.45f;
        m_OrientationCube = Instantiate(Resources.Load<GameObject>("OrientationCube"), oCubePos, Quaternion.identity);
        m_OrientationCube.transform.SetParent(transform.parent);
        UpdateOrientationCube();

        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(chest);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(head);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(armL);
        m_JdController.SetupBodyPart(forearmL);
        m_JdController.SetupBodyPart(handL);
        m_JdController.SetupBodyPart(armR);
        m_JdController.SetupBodyPart(forearmR);
        m_JdController.SetupBodyPart(handR);

        m_HipsRb = hips.GetComponent<Rigidbody>();
        m_ChestRb = chest.GetComponent<Rigidbody>();
        m_SpineRb = spine.GetComponent<Rigidbody>();

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Is this bp touching the ground
        
//        //RELATIVE RB VELOCITY
//        var velocityRelativeToLookRotationToTarget = m_worldPosMatrix.inverse.MultiplyVector(bp.rb.velocity);
//        sensor.AddObservation(velocityRelativeToLookRotationToTarget);
//
//        //RELATIVE RB ANGULAR VELOCITY
//        var angularVelocityRelativeToLookRotationToTarget = m_worldPosMatrix.inverse.MultiplyVector(bp.rb.angularVelocity);
//        sensor.AddObservation(angularVelocityRelativeToLookRotationToTarget);

        //RELATIVE RB VELOCITIES --WAS
//        sensor.AddObservation(m_OrientationCube.transform.InverseTransformVector(bp.rb.velocity));
//        sensor.AddObservation(m_OrientationCube.transform.InverseTransformVector(bp.rb.angularVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity)); //best if cube fixed rot?
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity)); //best if cube fixed rot?
//        sensor.AddObservation(bp.rb.velocity - m_JdController.bodyPartsDict[hips].rb.velocity);
//        sensor.AddObservation(bp.rb.angularVelocity - m_JdController.bodyPartsDict[hips].rb.angularVelocity);
        
        
        
//        sensor.AddObservation(bp.rb.velocity);
//        sensor.AddObservation(bp.rb.angularVelocity);
//        var localPosRelToHips = hips.InverseTransformPoint(rb.position);
//        sensor.AddObservation(localPosRelToHips);
//        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPointUnscaled(bp.rb.position));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));  //best
//        sensor.AddObservation(hips.InverseTransformPointUnscaled(bp.rb.position));
        

//        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR &&
//            bp.rb.transform != footL && bp.rb.transform != footR && bp.rb.transform != head)
        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR &&
            bp.rb.transform != footL && bp.rb.transform != footR)
        {
            sensor.AddObservation(RagdollHelpers.GetJointRotation(bp.joint));
//            sensor.AddObservation(bp.currentXNormalizedRot);
//            sensor.AddObservation(bp.currentYNormalizedRot);
//            sensor.AddObservation(bp.currentZNormalizedRot);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }
    
//    /// <summary>
//    /// Add relevant information on each body part to observations.
//    /// </summary>
//    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
//    {
//        var rb = bp.rb;
//        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Is this bp touching the ground
//        sensor.AddObservation(rb.velocity);
//        sensor.AddObservation(rb.angularVelocity);
//        var localPosRelToHips = hips.InverseTransformPoint(rb.position);
//        sensor.AddObservation(localPosRelToHips);
//
//        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR &&
//            bp.rb.transform != footL && bp.rb.transform != footR && bp.rb.transform != head)
//        {
//            sensor.AddObservation(bp.currentXNormalizedRot);
//            sensor.AddObservation(bp.currentYNormalizedRot);
//            sensor.AddObservation(bp.currentZNormalizedRot);
//            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
//        }
//    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
//        m_JdController.GetCurrentJointForces();
        
        // Update pos to target
//        m_WalkDir = target.position - hips.position;
//        m_WalkDir = target.position - m_OrientationCube.transform.position;
        

        sensor.AddObservation(RagdollHelpers.GetRotationDelta(m_WalkDirLookRot, hips.rotation));
//        sensor.AddObservation(RagdollHelpers.GetRotationDelta(m_WalkDirLookRot, chest.rotation));
        sensor.AddObservation(RagdollHelpers.GetRotationDelta(m_WalkDirLookRot, head.rotation));
//        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);


        
//        //HIP RAYCAST FOR HEIGHT
//        RaycastHit hit;
//        if (Physics.Raycast(hips.position, Vector3.down, out hit, 10.0f))
//        {
//            sensor.AddObservation(hit.distance);
//        }
//        else
//            sensor.AddObservation(10.0f);
        
//        //ORIENTATION MATRIX
//        Vector3 worldPosMatrixPos = hips.position;
//        worldPosMatrixPos.y = .5f;
//        m_worldPosMatrix  = Matrix4x4.TRS(worldPosMatrixPos, Quaternion.identity, Vector3.one);
        
//        sensor.AddObservation(m_WalkDir.normalized);

        //HIP POS REL TO MATRIX
//        sensor.AddObservation(hips.position - worldPosMatrixPos);
//        sensor.AddObservation(hips.position - m_OrientationCube.transform.position);
//        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.position);
        
//        sensor.AddObservation(hips.forward);
//        sensor.AddObservation(hips.up);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }

////        print(m_OrientationCube.transform.rotation.eulerAngles);
////        Debug.DrawRay(m_OrientationCube.transform.position, m_OrientationCube.transform.InverseTransformVector(m_JdController.bodyPartsDict[hips].rb.velocity), Color.green,Time.fixedDeltaTime * 5);
//  AddReward(
////            runForwardTowardsTargetReward
////            facingReward * velReward //max reward is moving towards while facing otherwise it is a penalty
////            +0.02f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity)
////            + 0.02f * Vector3.Dot(m_OrientationCube.transform.forward,Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity,5))
//            +0.01f * Vector3.Dot(m_OrientationCube.transform.forward,
//                Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity, 3))
//            + 0.01f * Vector3.Dot(m_OrientationCube.transform.forward, hips.forward)
//
////            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, chest.rotation) //reward looking at
////            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) //reward looking at
////            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) //reward looking at
////            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) - 1) *
////            .5f //penalize not looking at
////            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) - 1) *
////            .5f //penalize not looking at
//
//            + 0.005f * (head.position.y - shinL.position.y)
//            + 0.005f * (head.position.y - shinR.position.y)
////            + 0.01f * (head.position.y - shinL.position.y)
////            + 0.01f * (head.position.y - shinR.position.y)
////            - 0.005f * Mathf.Clamp(m_JdController.bodyPartsDict[handL].rb.velocity.magnitude,
////                6, 9999)
////            - 0.005f * Mathf.Clamp(m_JdController.bodyPartsDict[handR].rb.velocity.magnitude,
////                6, 9999)
////            + 0.02f * (head.position.y - hips.position.y)
////            - 0.01f * Vector3.Distance(m_JdController.bodyPartsDict[head].rb.velocity,
////                m_JdController.bodyPartsDict[hips].rb.velocity)
//        );

    }

    public override void OnActionReceived(float[] vectorAction)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var i = -1;

        bpDict[chest].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
        bpDict[spine].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);

        bpDict[thighL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[thighR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[shinL].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[shinR].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
        bpDict[footL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);


        bpDict[armL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[armR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
        bpDict[forearmL].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[forearmR].SetJointTargetRotation(vectorAction[++i], 0, 0);
        bpDict[head].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);

        //update joint strength settings
        bpDict[chest].SetJointStrength(vectorAction[++i]);
        bpDict[spine].SetJointStrength(vectorAction[++i]);
        bpDict[head].SetJointStrength(vectorAction[++i]);
        bpDict[thighL].SetJointStrength(vectorAction[++i]);
        bpDict[shinL].SetJointStrength(vectorAction[++i]);
        bpDict[footL].SetJointStrength(vectorAction[++i]);
        bpDict[thighR].SetJointStrength(vectorAction[++i]);
        bpDict[shinR].SetJointStrength(vectorAction[++i]);
        bpDict[footR].SetJointStrength(vectorAction[++i]);
        bpDict[armL].SetJointStrength(vectorAction[++i]);
        bpDict[forearmL].SetJointStrength(vectorAction[++i]);
        bpDict[armR].SetJointStrength(vectorAction[++i]);
        bpDict[forearmR].SetJointStrength(vectorAction[++i]);
//        print(Vector3.Dot(m_OrientationCube.transform.forward,
//            Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity, 3)));
//        print((Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) - 1) * .5f);
//        print(Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation));
//        print(Vector3.Dot(m_OrientationCube.transform.forward, hips.transform.forward));
    }

    void UpdateOrientationCube()
    {
        //FACING DIR
        m_WalkDir = target.position - m_OrientationCube.transform.position;
        m_WalkDir.y = 0;
//        m_WalkDir = walkDirWorldspace;
        m_WalkDirLookRot = Quaternion.LookRotation(m_WalkDir);
        
        
        //UPDATE ORIENTATION CUBE POS & ROT
        Vector3 oCubePos = hips.position;
//        oCubePos.y = -.45f;
        m_OrientationCube.transform.position = oCubePos;
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
        }
        UpdateOrientationCube();
        //reward looking at
//        float facingReward = + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation)
//                             + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation);

//        print($"FacingRewardDot {facingReward}");
//        float velReward = +0.02f * Vector3.Dot(m_OrientationCube.transform.forward,m_OrientationCube.transform.InverseTransformVector(m_JdController.bodyPartsDict[hips].rb.velocity));
//        print($"VelRewardDot {velReward}");
//        float velReward = +0.02f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity);





//        //Multiplying these amplifies the reward.
//        float facingReward = + 0.1f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation)
//                             + 0.1f * Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation);
//        float velReward = +0.2f * Vector3.Dot(m_OrientationCube.transform.forward,m_JdController.bodyPartsDict[hips].rb.velocity); //because we are observing in local space???
//        float runForwardTowardsTargetReward = facingReward * Mathf.Clamp(velReward, 0, 15);
        
//        print(Quaternion.Angle(hips.transform.rotation, thighL.transform.rotation));


//        print($"Combined {runForwardTowardsTargetReward}");
//        float runBackwardsTowardsTargetReward = facingReward * Mathf.Clamp(velReward, -1, 0);
        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        // d. Discourage head movement.
        AddReward(
//            runForwardTowardsTargetReward
//            facingReward * velReward //max reward is moving towards while facing otherwise it is a penalty
//            +0.02f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity)
//            + 0.02f * Vector3.Dot(m_OrientationCube.transform.forward,Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity,5))
            +0.01f * Vector3.Dot(m_OrientationCube.transform.forward,
                Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity, 5))
            + 0.01f * Vector3.Dot(m_OrientationCube.transform.forward, hips.forward)
            + 0.01f * Vector3.Dot(m_OrientationCube.transform.forward, hips.forward)

//            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, chest.rotation) //reward looking at
//            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) //reward looking at
//            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) //reward looking at
//            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) - 1) *
//            .5f //penalize not looking at
//            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) - 1) *
//            .5f //penalize not looking at

            + 0.005f * (head.position.y - shinL.position.y)
            + 0.005f * (head.position.y - shinR.position.y)
//            + 0.01f * (head.position.y - shinL.position.y)
//            + 0.01f * (head.position.y - shinR.position.y)
//            - 0.005f * Mathf.Clamp(m_JdController.bodyPartsDict[handL].rb.velocity.magnitude,
//                6, 9999)
//            - 0.005f * Mathf.Clamp(m_JdController.bodyPartsDict[handR].rb.velocity.magnitude,
//                6, 9999)
//            + 0.02f * (head.position.y - hips.position.y)
//            - 0.01f * Vector3.Distance(m_JdController.bodyPartsDict[head].rb.velocity,
//                m_JdController.bodyPartsDict[hips].rb.velocity)
        );
//        var handLVel = m_JdController.bodyPartsDict[handL].rb.velocity.magnitude;
//        var handRVel = m_JdController.bodyPartsDict[handR].rb.velocity.magnitude;
//        if (handLVel > 6)
//        {
//            AddReward(-0.005f * handLVel);
//        }
//        if (handRVel > 6)
//        {
//            AddReward(-0.005f * handRVel);
//        }
        
        
//        //SUNDAY VERSION
//        AddReward(
////            runForwardTowardsTargetReward
////            facingReward * velReward //max reward is moving towards while facing otherwise it is a penalty
////            +0.02f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity)
////            + 0.02f * Vector3.Dot(m_OrientationCube.transform.forward,Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity,5))
//            + 0.01f * Vector3.Dot(m_OrientationCube.transform.forward,Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity,3))
////            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) //reward looking at
////            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) //reward looking at
//            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation) - 1) * .5f //penalize not looking at
//            + 0.015f * (Quaternion.Dot(m_OrientationCube.transform.rotation, head.rotation) - 1) * .5f //penalize not looking at
//
//            
//            
////            + 0.02f * (head.position.y - hips.position.y)
////            - 0.01f * Vector3.Distance(m_JdController.bodyPartsDict[head].rb.velocity,
////                m_JdController.bodyPartsDict[hips].rb.velocity)
//        );
        
        
        
        
//        // Set reward for this step according to mixture of the following elements.
//        // a. Velocity alignment with goal direction.
//        // b. Rotation alignment with goal direction.
//        // c. Encourage head height.
//        // d. Discourage head movement.
//        m_WalkDir = target.position - m_OrientationCube.transform.position;
//        AddReward(
//            +0.03f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity)
//            + 0.01f * Quaternion.Dot(m_OrientationCube.transform.rotation, hips.rotation)
//            + 0.02f * (head.position.y - hips.position.y)
//            - 0.01f * Vector3.Distance(m_JdController.bodyPartsDict[head].rb.velocity,
//                m_JdController.bodyPartsDict[hips].rb.velocity)
//        );
//        m_WalkDir = target.position - m_JdController.bodyPartsDict[hips].rb.position;
//        AddReward(
//            +0.03f * Vector3.Dot(m_WalkDir.normalized, m_JdController.bodyPartsDict[hips].rb.velocity)
//            + 0.01f * Vector3.Dot(m_WalkDir.normalized, hips.forward)
//            + 0.02f * (head.position.y - hips.position.y)
//            - 0.01f * Vector3.Distance(m_JdController.bodyPartsDict[head].rb.velocity,
//                m_JdController.bodyPartsDict[hips].rb.velocity)
//        );
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

    
    
    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
//        print("OnEpisodeBegin");
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
//        if (m_WalkDir != Vector3.zero)
//        {
//            transform.rotation = Quaternion.LookRotation(m_WalkDir);
//        }
        transform.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
        UpdateOrientationCube();

//        transform.Rotate(Vector3.up, Random.Range(0.0f, 360.0f));

        if (detectTargets && !targetIsStatic)
        {
            GetRandomTargetPos();
        }

        SetResetParameters();
    }

    public void SetTorsoMass()
    {
        m_ChestRb.mass = m_ResetParams.GetWithDefault("chest_mass", 8);
        m_SpineRb.mass = m_ResetParams.GetWithDefault("spine_mass", 10);
        m_HipsRb.mass = m_ResetParams.GetWithDefault("hip_mass", 15);
    }

    public void SetResetParameters()
    {
        SetTorsoMass();
    }
}
