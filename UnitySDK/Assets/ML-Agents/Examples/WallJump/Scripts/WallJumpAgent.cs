//Put this script on your blue cube.

using System;
using System.Collections;
using UnityEngine;
using MLAgents;
using Random = UnityEngine.Random;

[RequireComponent(typeof(AgentCubeGroundCheck))] // Required for groundcheck
public class WallJumpAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;
    // Brain to use when no wall is present
    public Brain noWallBrain;
    // Brain to use when a jumpable wall is present
    public Brain smallWallBrain;
    // Brain to use when a wall requiring a block to jump over is present
    public Brain bigWallBrain;

    public GameObject ground;
    public GameObject spawnArea;
    public GameObject shortBlock;
    public Rigidbody hazardRb;
    public GameObject wall;
    Bounds m_SpawnAreaBounds;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    WallJumpAcademy m_Academy;
    RayPerception m_RayPer;
    string[] m_DetectableObjects;

    //Groundcheck
    AgentCubeGroundCheck m_groundCheck;
    AgentCubeMovement m_agentMovement;


    public override void InitializeAgent()
    {
        m_groundCheck = GetComponent<AgentCubeGroundCheck>();
        m_Academy = FindObjectOfType<WallJumpAcademy>();
        m_agentMovement = FindObjectOfType<AgentCubeMovement>();
        m_RayPer = GetComponent<RayPerception>();
        m_Configuration = Random.Range(0, 5);
        m_DetectableObjects = new[] { "wall", "goal", "block", "hazard" };
        m_AgentRb = GetComponent<Rigidbody>();
        m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        spawnArea.SetActive(false);
    }

    public override void CollectObservations()
    {
        var rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 0f, 0f));
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 2.5f, 2.5f));
        var agentPos = m_AgentRb.position - ground.transform.position;

        AddVectorObs(agentPos / 20f); //help with orientation
        AddVectorObs(m_groundCheck.isGrounded);
        AddVectorObs(m_AgentRb.velocity/m_agentMovement.agentRunSpeed); //normalized vel
        AddVectorObs(m_AgentRb.angularVelocity/m_AgentRb.maxAngularVelocity); //normalized angVel
        AddVectorObs(m_AgentRb.transform.forward); //help with orientation

    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    public Vector3 GetRandomSpawnPos()
    {
        var randomPosX = Random.Range(-m_SpawnAreaBounds.extents.x,
            m_SpawnAreaBounds.extents.x);
        var randomPosZ = Random.Range(-m_SpawnAreaBounds.extents.z,
            m_SpawnAreaBounds.extents.z);

        var randomSpawnPos = spawnArea.transform.position +
            new Vector3(randomPosX, 0.45f, randomPosZ);
        return randomSpawnPos;
    }

    /// <summary>
    /// Changes the color of the ground for a moment
    /// </summary>
    /// <returns>The Enumerator to be used in a Coroutine</returns>
    /// <param name="mat">The material to be swaped.</param>
    /// <param name="time">The time the material will remain.</param>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(float[] act)
    {
        AddReward(-0.0005f); //hurry up

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var dirToGoForwardAction = (int)act[0];
        var rotateDirAction = (int)act[1];
        var jumpAction = (int)act[2];
        if (dirToGoForwardAction == 1)
            dirToGo += transform.forward;
        else if (dirToGoForwardAction == 2)
            dirToGo += -transform.forward;
        if (rotateDirAction == 1)
            rotateDir = -transform.up ;
        else if (rotateDirAction == 2)
            rotateDir = transform.up;

        
        
        
        
        //handle jumping
        if (jumpAction == 1)
        {
            if (m_groundCheck.isGrounded)
            {
                m_agentMovement.Jump(m_AgentRb);
            }
        }

        //handle body rotation
        if (rotateDir != Vector3.zero)
        {
            m_agentMovement.RotateBody(m_AgentRb, rotateDir);
        }

        //handle running        
        if (dirToGo != Vector3.zero)
        {
            if (!m_groundCheck.isGrounded)
            {
                m_agentMovement.RunInAir(m_AgentRb, dirToGo.normalized);
            }
            else
            {
                m_agentMovement.RunOnGround(m_AgentRb, dirToGo.normalized);
            }
        }
        
        //handle idle drag
        if (m_groundCheck.isGrounded && dirToGo == Vector3.zero)
        {
            m_agentMovement.AddIdleDrag(m_AgentRb);
        }

        //handle falling forces
        if (!m_groundCheck.isGrounded)
        {
            m_agentMovement.AddFallingForce(m_AgentRb);
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (IsDone())
        {
            return;
        }
        MoveAgent(vectorAction);
        if(m_AgentRb.position.y < -1 || m_ShortBlockRb.position.y < -1 || hazardRb.position.y < -1)
        {
            SetReward(-1f);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
            Done();
        }
    }



    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("hazard"))
        {
            SetReward(-1f);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
            Done();
        }
    }

    // Detect when the agent hits the goal
    void OnTriggerStay(Collider col)
    {
        if (IsDone())
        {
            return;
        }
        if (col.gameObject.CompareTag("goal") && m_groundCheck.isGrounded)
        {
            SetReward(1f);
//            ResetBlock(m_ShortBlockRb);
//            ResetBlock(hazardRb);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.goalScoredMaterial, .5f));
            Done();
        }
    }

    //Reset the orange block position
    void ResetBlock(Rigidbody blockRb)
    {
        blockRb.transform.position = GetRandomSpawnPos();
        blockRb.velocity = Vector3.zero;
        blockRb.angularVelocity = Vector3.zero;
    }

    public override void AgentReset()
    {
        ResetBlock(m_ShortBlockRb);
        ResetBlock(hazardRb);
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, -12);
        m_Configuration = Random.Range(0, 5);
        m_AgentRb.velocity = Vector3.zero;
    }

//    private void FixedUpdate()
//    {
//        if (m_Configuration != -1)
//        {
//            ConfigureAgent(m_Configuration);
//            m_Configuration = -1;
//        }
//    }

    /// <summary>
    /// Configures the agent. Given an integer config, the wall will have
    /// different height and a different brain will be assigned to the agent.
    /// </summary>
    /// <param name="config">Config.
    /// If 0 : No wall and noWallBrain.
    /// If 1:  Small wall and smallWallBrain.
    /// Other : Tall wall and BigWallBrain. </param>
    void ConfigureAgent(int config)
    {
        var localScale = wall.transform.localScale;
        if (config == 0)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.resetParameters["no_wall_height"],
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(noWallBrain);
        }
        else if (config == 1)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.resetParameters["small_wall_height"],
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(smallWallBrain);
        }
        else
        {
            var height =
                m_Academy.resetParameters["big_wall_min_height"] +
                Random.value * (m_Academy.resetParameters["big_wall_max_height"] -
                    m_Academy.resetParameters["big_wall_min_height"]);
            localScale = new Vector3(
                localScale.x,
                height,
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(bigWallBrain);
        }
    }
}
