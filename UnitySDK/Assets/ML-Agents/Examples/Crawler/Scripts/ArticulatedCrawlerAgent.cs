using System;
using UnityEngine;
using MLAgents;
using Random = UnityEngine.Random;

[RequireComponent(typeof(ArticulatedJointDriveController))] // Required to set joint forces
public class ArticulatedCrawlerAgent : Agent
{
    [Header("Target To Walk Towards")][Space(10)]
    public Transform target;

    public Transform ground;
    public bool detectTargets;
    public bool targetIsStatic = false;
    public bool respawnTargetWhenTouched;
    public float targetSpawnRadius;

    [Header("Body Parts")][Space(10)] 
    public Transform rootBodyPrefab; 
    public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    private string bodyName;
    private string leg0UpperName;
    private string leg0LowerName;
    private string leg1UpperName;
    private string leg1LowerName;
    private string leg2UpperName;
    private string leg2LowerName;
    private string leg3UpperName;
    private string leg3LowerName;
    
    
    [Header("Joint Settings")][Space(10)] ArticulatedJointDriveController m_JdController;
    Vector3 m_DirToTarget;
    float m_MovingTowardsDot;
    float m_FacingDot;

    [Header("Reward Functions To Use")][Space(10)]
    public bool rewardMovingTowardsTarget; // Agent should move towards target

    public bool rewardFacingTarget; // Agent should face the target
    public bool rewardUseTimePenalty; // Hurry up

    [Header("Foot Grounded Visualization")][Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;
    bool m_IsNewDecisionStep;
    int m_CurrentDecisionStep;

    Quaternion m_LookRotation;
    Matrix4x4 m_TargetDirMatrix;

    public override void InitializeAgent()
    {
        m_JdController = GetComponent<ArticulatedJointDriveController>();
        m_CurrentDecisionStep = 1;
        m_DirToTarget = target.position - body.position;

        m_JdController.Reset();
        SetupBodyParts();
        SaveBodyPartNames();
    }

    /// <summary>
    /// Setup body parts - add body part and configure ArticulatedJoinDriveController
    /// </summary>
    private void SetupBodyParts()
    {
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

    private void SaveBodyPartNames()
    {
        bodyName = body.name;
        leg0UpperName = leg0Upper.name;
        leg0LowerName = leg0Lower.name;
        leg1UpperName = leg1Upper.name;
        leg1LowerName = leg1Lower.name;
        leg2UpperName = leg2Upper.name;
        leg2LowerName = leg2Lower.name;
        leg3UpperName = leg3Upper.name;
        leg3LowerName = leg3Lower.name;
    }
    
    /// <summary>
    /// We only need to change the joint settings based on decision freq.
    /// </summary>
    public void IncrementDecisionTimer()
    {
        if (m_CurrentDecisionStep == agentParameters.numberOfActionsBetweenDecisions
            || agentParameters.numberOfActionsBetweenDecisions == 1)
        {
            m_CurrentDecisionStep = 1;
            m_IsNewDecisionStep = true;
        }
        else
        {
            m_CurrentDecisionStep++;
            m_IsNewDecisionStep = false;
        }
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(ArticulationBodyPart bp)
    {
        var arb = bp.arb;
        AddVectorObs(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground

        var velocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(arb.velocity);
        AddVectorObs(velocityRelativeToLookRotationToTarget);

        var angularVelocityRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(arb.angularVelocity);
        AddVectorObs(angularVelocityRelativeToLookRotationToTarget);

        if (bp.arb.transform != body)
        {
            var localPosRelToBody = body.InverseTransformPoint(arb.transform.position); // Translate from world space to body local space, since all articulations are children of body in hiearchy
            AddVectorObs(localPosRelToBody);
            AddVectorObs(bp.currentXNormalizedRot); // Current x rot
            AddVectorObs(bp.currentYNormalizedRot); // Current y rot
            AddVectorObs(bp.currentZNormalizedRot); // Current z rot
            AddVectorObs(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public override void CollectObservations()
    {
        // Update pos to target
        m_DirToTarget = target.position - body.position;
        m_LookRotation = Quaternion.LookRotation(m_DirToTarget);
        m_TargetDirMatrix = Matrix4x4.TRS(Vector3.zero, m_LookRotation, Vector3.one);

        RaycastHit hit;
        if (Physics.Raycast(body.position, Vector3.down, out hit, 10.0f))
        {
            AddVectorObs(hit.distance);
        }
        else
            AddVectorObs(10.0f);

        // Forward & up to help with orientation
        var bodyForwardRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(body.forward);
        AddVectorObs(bodyForwardRelativeToLookRotationToTarget);

        var bodyUpRelativeToLookRotationToTarget = m_TargetDirMatrix.inverse.MultiplyVector(body.up);
        AddVectorObs(bodyUpRelativeToLookRotationToTarget);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
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
        newTargetPos.y = 0.5f;
        target.position = newTargetPos + ground.position;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (detectTargets)
        {
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                if (bodyPart.targetContact && !IsDone() && bodyPart.targetContact.touchingTarget)
                {
                    TouchedTarget();
                }
            }
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

        // Joint update logic only needs to happen when a new decision is made
        if (m_IsNewDecisionStep)
        {
            // The dictionary with all the body parts in it are in the jdController
            var bpDict = m_JdController.bodyPartsDict;

            var i = -1;
            // Pick a new target joint rotation
            bpDict[leg0Upper].SetJointTargetRotation(0, vectorAction[++i], vectorAction[++i]);
            bpDict[leg1Upper].SetJointTargetRotation(0, vectorAction[++i], vectorAction[++i]);
            bpDict[leg2Upper].SetJointTargetRotation(0, vectorAction[++i], vectorAction[++i]);
            bpDict[leg3Upper].SetJointTargetRotation(0, vectorAction[++i], vectorAction[++i]);
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

        IncrementDecisionTimer();
    }

    /// <summary>
    /// Reward moving towards target & Penalize moving away from target.
    /// </summary>
    void RewardFunctionMovingTowards()
    {
        m_MovingTowardsDot = Vector3.Dot(m_JdController.bodyPartsDict[body].arb.velocity, m_DirToTarget.normalized);
        AddReward(0.03f * m_MovingTowardsDot);
    }

    /// <summary>
    /// Reward facing target & Penalize facing away from target
    /// </summary>
    void RewardFunctionFacingTarget()
    {
        m_FacingDot = Vector3.Dot(m_DirToTarget.normalized, body.forward);
        AddReward(0.01f * m_FacingDot);
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
    public override void AgentReset()
    {
        Vector3 position = m_JdController.bodyPartsDict[body].startingPos;
        
        // For starting position, make a random orientation
        Quaternion rotation = Quaternion.identity;
        Vector3 randomViewPos = Random.onUnitSphere * 10.0f + position;
        randomViewPos.y = position.y; // Look at the height of body center
        
        rotation.SetLookRotation(randomViewPos, Vector3.up);
        
        m_JdController.Reset();
        
        string bodyName = body.name;
        DestroyImmediate(body.gameObject);
        body = Instantiate(rootBodyPrefab, position, rotation);
        body.transform.parent = transform;
        body.name = bodyName;

        
        ResetLegTransforms(body);
        SetupBodyParts();
        
        if (!targetIsStatic)
        {
            GetRandomTargetPos();
        }
        m_IsNewDecisionStep = true;
        m_CurrentDecisionStep = 1;
    }
    /// <summary>
    /// After spawning new prefab, reinitialize transforms of body parts
    /// </summary>
    /// <param name="rootBody"></param>
    private void ResetLegTransforms(Transform rootBody)
    {
        leg0Upper = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg0UpperName);
        leg1Upper = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg1UpperName);
        leg2Upper = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg2UpperName);
        leg3Upper = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg3UpperName);

        leg0Lower = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg0LowerName);
        leg1Lower = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg1LowerName);
        leg2Lower = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg2LowerName);
        leg3Lower = ArticulatedJointDriveController.FindBodyPartByName(rootBody, leg3LowerName);
    }
}
