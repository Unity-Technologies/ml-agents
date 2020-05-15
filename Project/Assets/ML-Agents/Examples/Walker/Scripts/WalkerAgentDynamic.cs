using MLAgentsExamples;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using UnityEditor;
using BodyPart = Unity.MLAgentsExamples.BodyPart;

public class WalkerAgentDynamic : Agent
{
    [Header("Walking Speed")]
    [Space(10)]
    [Header("Specific to Walker")]
    public float maximumWalkingSpeed = 999; //The max walk velocity magnitude an agent will be rewarded for
    Vector3 m_WalkDir;
    Quaternion m_WalkDirLookRot;
    
    [Space(10)]
    [Header("Orientation Cube")]
    [Space(10)]
    //This will be used as a stable observation platform for the ragdoll to use.
    GameObject m_OrientationCube;
    public Transform directionIndicator;
    
    [Header("Target To Walk Towards")]
    [Space(10)]
    public float targetSpawnRadius;
    public Transform target;
    public Transform ground;
    public bool detectTargets;
    public bool targetIsStatic;
    public bool respawnTargetWhenTouched;
    

    [Header("Body Parts")]
    [Space(10)]
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
    


    public override void Initialize()
    {
        //Spawn an orientation cube
        Vector3 oCubePos = hips.position;
        oCubePos.y = -.45f;
        m_OrientationCube = Instantiate(Resources.Load<GameObject>("OrientationCube"), oCubePos, Quaternion.identity);
        m_OrientationCube.transform.SetParent(transform);
        
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
        
        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));
        
        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));  //best

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }
    
    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, m_OrientationCube.transform.forward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, m_OrientationCube.transform.forward));

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
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
    }

    void UpdateOrientationCube()
    {
        //FACING DIR
        m_WalkDir = target.position - m_OrientationCube.transform.position;
        m_WalkDir.y = 0; //flatten dir on the y
        m_WalkDirLookRot = Quaternion.LookRotation(m_WalkDir); //get our look rot to the target
        
        //UPDATE ORIENTATION CUBE POS & ROT
        m_OrientationCube.transform.position = hips.position;
        m_OrientationCube.transform.rotation = m_WalkDirLookRot;
        
        directionIndicator.position = new Vector3(hips.position.x, directionIndicator.position.y, hips.position.z);
        directionIndicator.rotation = m_WalkDirLookRot;
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
        
        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        AddReward(
            +0.02f * Vector3.Dot(m_OrientationCube.transform.forward,
                Vector3.ClampMagnitude(m_JdController.bodyPartsDict[hips].rb.velocity, maximumWalkingSpeed))
            + 0.01f * Vector3.Dot(m_OrientationCube.transform.forward, head.forward)
            + 0.005f * (head.position.y - footL.position.y)
            + 0.005f * (head.position.y - footR.position.y)
        );
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
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
        
        //Random start rotation
        transform.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
        
        UpdateOrientationCube();

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
