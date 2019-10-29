//Put this script on your blue cube.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using Random = UnityEngine.Random;

[RequireComponent(typeof(AgentCubeGroundCheck))] // Required for groundcheck
public class BuilderAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    BuilderAcademy m_Academy;
    RayPerception m_RayPer;
    string[] m_DetectableObjects;
    Rigidbody m_AgentRb;
    public List<Rigidbody> buildingBlocksList = new List<Rigidbody>();

    //Groundcheck
    AgentCubeGroundCheck m_groundCheck;
    AgentCubeMovement m_agentMovement;
    
    public bool grabbingItem;
    public Rigidbody grabbedItemRb;
    public Collider grabbedItemCol;
    private Transform m_AreaTransform;
    public override void InitializeAgent()
    {
        m_AreaTransform = transform.parent;
        m_groundCheck = GetComponent<AgentCubeGroundCheck>();
        m_Academy = FindObjectOfType<BuilderAcademy>();
        m_agentMovement = FindObjectOfType<AgentCubeMovement>();
        m_RayPer = GetComponent<RayPerception>();
        m_Configuration = Random.Range(0, 5);
        m_DetectableObjects = new[] { "block" };
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_AgentRb = GetComponent<Rigidbody>();
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
            rayDistance, rayAngles, m_DetectableObjects, 1.5f, 0f));
//        AddVectorObs(m_RayPer.Perceive(
//            rayDistance, rayAngles, m_DetectableObjects, 2.5f, 2.5f));
        var agentPos = m_AgentRb.position - ground.transform.position;

        AddVectorObs(agentPos / 20f); //help with orientation
        AddVectorObs(m_groundCheck.isGrounded);
        AddVectorObs(m_AgentRb.velocity/m_agentMovement.agentRunSpeed); //normalized vel
        AddVectorObs(m_AgentRb.angularVelocity/m_AgentRb.maxAngularVelocity); //normalized angVel
        AddVectorObs(m_AgentRb.transform.forward); //help with orientation
                
        AddVectorObs(grabbingItem);
        foreach (var item in buildingBlocksList)
        {
            var blockPos = item.position - ground.transform.position;
            AddVectorObs(blockPos / 20f); //help with orientation
        }
    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    public Vector3 GetRandomSpawnPos()
    {
        var randomPosX = Random.Range(-m_SpawnAreaBounds.extents.x,
            m_SpawnAreaBounds.extents.x);
        var randomPosY = Random.Range(-m_SpawnAreaBounds.extents.y,
            m_SpawnAreaBounds.extents.y);
        var randomPosZ = Random.Range(-m_SpawnAreaBounds.extents.z,
            m_SpawnAreaBounds.extents.z);

        var randomSpawnPos = spawnArea.transform.position +
            new Vector3(randomPosX, randomPosY, randomPosZ);
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
        
        
                
        
        
        
        
        var grabOrReleaseAction = (int)act[3];
        if (grabOrReleaseAction == 1)
        {
            if(grabbingItem)
            {
                ReleaseBlock();
            }
            else
            {
                GrabBlock();
            }
        }

        
        
        
        
        
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

    bool AllBlockOnPlatform()
    {
        foreach (var item in buildingBlocksList)
        {
            if (item.position.y < ground.transform.position.y -1)
            {
                return false;
            }
        }
        return true;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.G))
        {
            if(grabbingItem)
            {
                ReleaseBlock();
            }
            else
            {
                GrabBlock();
            }
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (IsDone())
        {
            return;
        }
        MoveAgent(vectorAction);
        if (m_AgentRb.position.y < ground.transform.position.y -1|| !AllBlockOnPlatform())
        {
            SetReward(-1f);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
            Done();
        }

        if (m_groundCheck.isGrounded)
        {
            AddReward((m_AgentRb.transform.localPosition.y - 1) * m_Academy.heightRewardCoeff);
        }
    }

    void ResetAllBlocks()
    {
        foreach (var item in buildingBlocksList)
        {
            ResetBlock(item);
        }
    }
    void GrabBlock()
    {
        RaycastHit hit;
        if (Physics.Raycast(m_AgentRb.position, transform.forward, out hit, 1f))
        {
            if (hit.collider.gameObject.CompareTag("block"))
            {
                hit.rigidbody.isKinematic = true;
                hit.transform.SetParent(transform);
                grabbedItemRb = hit.transform.GetComponent<Rigidbody>();
                grabbedItemCol = hit.collider;
                hit.collider.enabled = false;
                grabbingItem = true;
                hit.transform.GetComponent<Renderer>().material = m_Academy.grabbedMaterial;
                print("GRABBED");
            }
        }
    }
    
    void ReleaseBlock()
    {
        if (grabbedItemRb)
        {
            ResetBlock(grabbedItemRb);
        }

                print("RELEASED");
        grabbedItemRb = null;
        grabbedItemCol = null;
        grabbingItem = false;
    }
//    void OnCollisionEnter(Collision col)
//    {
//        if (col.gameObject.CompareTag("hazard"))
//        {
//            SetReward(-1f);
//            StartCoroutine(
//                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
//            Done();
//        }
//    }

//    // Detect when the agent hits the goal
//    void OnTriggerStay(Collider col)
//    {
//        if (IsDone())
//        {
//            return;
//        }
//        if (col.gameObject.CompareTag("goal") && m_groundCheck.isGrounded)
//        {
//            SetReward(1f);
////            ResetBlock(m_ShortBlockRb);
////            ResetBlock(hazardRb);
//            StartCoroutine(
//                GoalScoredSwapGroundMaterial(m_Academy.goalScoredMaterial, .5f));
//            Done();
//        }
//    }

    //Reset the orange block position
    void ResetBlock(Rigidbody blockRb)
    {
        blockRb.transform.SetParent(m_AreaTransform);
        blockRb.velocity = Vector3.zero;
        blockRb.angularVelocity = Vector3.zero;
        blockRb.isKinematic = false;
        blockRb.transform.GetComponent<Renderer>().material = m_Academy.notGrabbedMaterial;

        blockRb.GetComponent<Collider>().enabled = true;

    }

    public override void AgentReset()
    {
        ResetAllBlocks();
        foreach (var item in buildingBlocksList)
        {
            item.transform.position = GetRandomSpawnPos();
        }
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, 0);
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

//    /// <summary>
//    /// Configures the agent. Given an integer config, the wall will have
//    /// different height and a different brain will be assigned to the agent.
//    /// </summary>
//    /// <param name="config">Config.
//    /// If 0 : No wall and noWallBrain.
//    /// If 1:  Small wall and smallWallBrain.
//    /// Other : Tall wall and BigWallBrain. </param>
//    void ConfigureAgent(int config)
//    {
//        var localScale = wall.transform.localScale;
//        if (config == 0)
//        {
//            localScale = new Vector3(
//                localScale.x,
//                m_Academy.resetParameters["no_wall_height"],
//                localScale.z);
//            wall.transform.localScale = localScale;
//            GiveBrain(noWallBrain);
//        }
//        else if (config == 1)
//        {
//            localScale = new Vector3(
//                localScale.x,
//                m_Academy.resetParameters["small_wall_height"],
//                localScale.z);
//            wall.transform.localScale = localScale;
//            GiveBrain(smallWallBrain);
//        }
//        else
//        {
//            var height =
//                m_Academy.resetParameters["big_wall_min_height"] +
//                Random.value * (m_Academy.resetParameters["big_wall_max_height"] -
//                    m_Academy.resetParameters["big_wall_min_height"]);
//            localScale = new Vector3(
//                localScale.x,
//                height,
//                localScale.z);
//            wall.transform.localScale = localScale;
//            GiveBrain(bigWallBrain);
//        }
//    }
}
