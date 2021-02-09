//Put this script on your blue cube.

using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Runtime.Input;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

public class PushAgentBasic : Agent, IInputActionAssetProvider
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;

    public GameObject area;

    /// <summary>
    /// The area bounds.
    /// </summary>
    [HideInInspector]
    public Bounds areaBounds;

    PushBlockSettings m_PushBlockSettings;

    /// <summary>
    /// The goal to push the block to.
    /// </summary>
    public GameObject goal;

    /// <summary>
    /// The block to be pushed to the goal.
    /// </summary>
    public GameObject block;

    public float JumpTime = 0.5f;
    float m_JumpTimeRemaining;

    /// <summary>
    /// Detects when the block touches the goal.
    /// </summary>
    [HideInInspector]
    public GoalDetect goalDetect;

    public bool useVectorObs;

    Rigidbody m_BlockRb;  //cached on initialization
    Rigidbody m_AgentRb;  //cached on initialization
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    EnvironmentParameters m_ResetParams;

    PushBlockActions m_PushblockActions;

    float m_JumpCoolDownStart;

    void Awake()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        m_PushblockActions = new PushBlockActions();
        m_PushblockActions.Enable();
        m_PushblockActions.Movement.jump.performed += JumpOnperformed;

        goalDetect = block.GetComponent<GoalDetect>();
        goalDetect.agent = this;

        // Cache the agent rigidbody
        m_AgentRb = GetComponent<Rigidbody>();
        // Cache the block rigidbody
        m_BlockRb = block.GetComponent<Rigidbody>();
        // Get the ground's bounds
        areaBounds = ground.GetComponent<Collider>().bounds;
        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;
    }

    void JumpOnperformed(InputAction.CallbackContext callbackContext)
    {
        InnerJump(gameObject.transform);
    }

    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();
    }

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        var tries = 0;
        while (foundNewSpawnLocation == false && tries < 50)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(2.5f, 0.01f, 2.5f)) == false)
            {
                foundNewSpawnLocation = true;
            }
            tries++;
        }
        return randomSpawnPos;
    }

    void FixedUpdate()
    {
        InnerMove(gameObject.transform, m_PushblockActions.Movement.movement.ReadValue<Vector2>());
        if (m_JumpTimeRemaining < 0)
        {
            m_AgentRb.AddForce(-transform.up * (m_PushBlockSettings.agentJumpForce * 3), ForceMode.Acceleration);
        }
        m_JumpTimeRemaining -= Time.fixedDeltaTime;
    }

    /// <summary>
    /// Called when the agent moves the block into the goal.
    /// </summary>
    public void ScoredAGoal()
    {
        // We use a reward of 5.
        AddReward(5f);

        // By marking an agent as done AgentReset() will be called automatically.
        EndEpisode();

        // Swap ground material for a bit to indicate we scored.
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));
    }

    /// <summary>
    /// Swap ground material, wait time seconds, then swap back to the regular material.
    /// </summary>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionBuffers actionBuffers)
    {
        var move = actionBuffers.ContinuousActions;
        var transform1 = transform;
        if (!move.IsEmpty())
        {
            var movement = new Vector2(move[0], move[1]);
            InnerMove(transform1, movement);
        }

        var jump = actionBuffers.DiscreteActions;
        if (!jump.IsEmpty())
        {
            if (jump[0] == 1)
            {
                InnerJump(transform1);
            }
        }
    }

    static float CreateUpVector(Vector2 move)
    {
        return Mathf.Abs(move.x) > Mathf.Abs(move.y) ? move.x : 0f;
    }

    static float CreateForwardVector(Vector2 move)
    {
        return Mathf.Abs(move.y) > Mathf.Abs(move.x) ? move.y : 0f;
    }

    void InnerJump(Transform t)
    {
        if (Time.realtimeSinceStartup - m_JumpCoolDownStart > m_PushBlockSettings.agentJumpCoolDown)
        {
            m_JumpTimeRemaining = JumpTime;
            m_AgentRb.AddForce(t.up * m_PushBlockSettings.agentJumpForce, ForceMode.VelocityChange);
            m_JumpCoolDownStart = Time.realtimeSinceStartup;
        }
    }

    void InnerMove(Transform t, Vector2 v)
    {
        var forward = CreateForwardVector(v);
        var up = CreateUpVector(v);
        var dirToGo = t.forward * forward;
        var rotateDir = t.up * up;
        t.Rotate(rotateDir, Time.deltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Penalty given each step to encourage agent to finish task quickly.
        AddReward(-1f / MaxStep);
    }

    /// <summary>
    /// Resets the block position and velocities.
    /// </summary>
    void ResetBlock()
    {
        // Get a random position for the block.
        block.transform.position = GetRandomSpawnPos();

        // Reset block velocity back to zero.
        m_BlockRb.velocity = Vector3.zero;

        // Reset block angularVelocity back to zero.
        m_BlockRb.angularVelocity = Vector3.zero;
    }

    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        area.transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        ResetBlock();
        transform.position = GetRandomSpawnPos();
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;

        SetResetParameters();
    }

    public void SetGroundMaterialFriction()
    {
        var groundCollider = ground.GetComponent<Collider>();

        groundCollider.material.dynamicFriction = m_ResetParams.GetWithDefault("dynamic_friction", 0);
        groundCollider.material.staticFriction = m_ResetParams.GetWithDefault("static_friction", 0);
    }

    public void SetBlockProperties()
    {
        var scale = m_ResetParams.GetWithDefault("block_scale", 2);
        //Set the scale of the block
        m_BlockRb.transform.localScale = new Vector3(scale, 0.75f, scale);

        // Set the drag of the block
        m_BlockRb.drag = m_ResetParams.GetWithDefault("block_drag", 0.5f);
    }

    void SetResetParameters()
    {
        SetGroundMaterialFriction();
        SetBlockProperties();
    }

    public InputActionAsset GetInputActionAsset()
    {
        return m_PushblockActions.asset;
    }
}
