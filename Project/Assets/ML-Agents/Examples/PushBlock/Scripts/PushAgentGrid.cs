//Put this script on your blue cube.

using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

using Random = UnityEngine.Random;

public class PushAgentGrid : Agent
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

    /// <summary>
    /// Detects when the block touches the goal.
    /// </summary>
    //    [HideInInspector]
    //    public GoalDetect goalDetect;

    public bool useVectorObs;

    Rigidbody m_BlockRb;  //cached on initialization
    Rigidbody m_AgentRb;  //cached on initialization
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    EnvironmentParameters m_ResetParams;
    public bool useAutoFocus = false;
    public List<RayPerceptionSensorComponent3D> raySensorsList = new List<RayPerceptionSensorComponent3D>();
    public float focusAngle = 90;

    public bool canLookAround;
    public float lookAngle = 0;
    public float maxLookAngle = 45f;

    //    public float rayHeight = 0;
    void Awake()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
    }

    public override void Initialize()
    {
        //        goalDetect = block.GetComponent<GoalDetect>();
        //        goalDetect.agent = this;

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

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVectorObs)
        {
            var localVelocity = transform.InverseTransformDirection(m_AgentRb.velocity);
            sensor.AddObservation(localVelocity.x);
            sensor.AddObservation(localVelocity.z);
            //            sensor.AddObservation(transform.localRotation);
            //            sensor.AddObservation(transform.localPosition);
            if (useAutoFocus)
            {
                sensor.AddObservation(focusAngle / 180);
                //                sensor.AddObservation(rayHeight);
            }
            if (canLookAround)
            {
                sensor.AddObservation(lookAngle / maxLookAngle);
                //                sensor.AddObservation(rayHeight);
            }
        }
    }

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            //            if (Physics.CheckBox(randomSpawnPos, new Vector3(2.5f, 0.01f, 2.5f)) == false)
            if (Physics.CheckBox(randomSpawnPos, new Vector3(.75f, 0.01f, .75f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
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

    //    /// <summary>
    //    /// Moves the agent according to the selected action.
    //    /// </summary>
    //    public void MoveAgent(ActionBuffers actionBuffers)
    //    {
    //        var dirToGo = Vector3.zero;
    //        var rotateDir = Vector3.zero;
    //
    ////        var forwardAction = act[0];
    ////        var lateralAction = act[1];
    ////        var rotatedAction = act[2];
    //        var act = actionBuffers.ContinuousActions;
    //        dirToGo += transform.forward * act[0];
    //        dirToGo += transform.right * act[1];
    //
    //        rotateDir = transform.up * act[2];
    ////        switch (action)
    ////        {
    ////            case 1:
    ////                dirToGo = transform.forward * 1f;
    ////                break;
    ////            case 2:
    ////                dirToGo = transform.forward * -1f;
    ////                break;
    ////            case 3:
    ////                rotateDir = transform.up * 1f;
    ////                break;
    ////            case 4:
    ////                rotateDir = transform.up * -1f;
    ////                break;
    ////            case 5:
    ////                dirToGo = transform.right * -0.75f;
    ////                break;
    ////            case 6:
    ////                dirToGo = transform.right * 0.75f;
    ////                break;
    ////        }
    //        transform.Rotate(rotateDir.normalized, Time.fixedDeltaTime * 200f);
    //        m_AgentRb.AddForce(dirToGo.normalized * m_PushBlockSettings.agentRunSpeed,
    //            ForceMode.VelocityChange);
    //
    //
    ////        if (canLookAround)
    ////        {
    ////            var angle = act[1];
    ////            switch (angle)
    ////            {
    ////                case 0:
    ////                    lookAngle = 0;
    ////                    break;
    ////                case 1:
    ////                    lookAngle = maxLookAngle/3;
    ////                    break;
    ////                case 2:
    ////                    lookAngle = maxLookAngle/2;
    ////                    break;
    ////                case 3:
    ////                    lookAngle = maxLookAngle/1;
    ////                    break;
    ////                case 4:
    ////                    lookAngle = -maxLookAngle/3;
    ////                    break;
    ////                case 5:
    ////                    lookAngle = -maxLookAngle/2;
    ////                    break;
    ////                case 6:
    ////                    lookAngle = -maxLookAngle/1;
    ////                    break;
    ////            }
    ////
    ////            foreach (var item in raySensorsList)
    ////            {
    ////                item.transform.localRotation = Quaternion.Euler(0,lookAngle,0);
    ////            }
    ////        }
    //
    //
    //        if (useAutoFocus)
    //        {
    //            focusAngle = Mathf.Lerp(0, 180, (act[3] + 1f) * 0.5f);
    ////            var focus = act[1];
    ////            var focusAngleTarget = act[1];
    ////            switch (focus)
    ////            {
    ////                case 0:
    ////                    focusAngle = 0;
    ////                    break;
    ////                case 1:
    ////                    focusAngle = 30;
    ////                    break;
    ////                case 2:
    ////                    focusAngle = 60;
    ////                    break;
    ////                case 3:
    ////                    focusAngle = 90;
    ////                    break;
    ////                case 4:
    ////                    focusAngle = 120;
    ////                    break;
    ////                case 5:
    ////                    focusAngle = 180;
    ////                    break;
    ////            }
    ////                    Mathf.MoveTowards()
    ////            var height = act[2];
    ////            switch (height)
    ////            {
    ////                case 0:
    ////                    rayHeight = 0;
    ////                    break;
    ////                case 1:
    ////                    rayHeight = 1;
    ////                    break;
    ////            }
    //
    //            foreach (var item in raySensorsList)
    //            {
    //                item.MaxRayDegrees = focusAngle;
    ////                item.StartVerticalOffset = rayHeight;
    ////                item.EndVerticalOffset = rayHeight;
    //            }
    //        }
    //    }


    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);


        if (canLookAround)
        {
            var angle = act[1];
            switch (angle)
            {
                case 0:
                    lookAngle = 0;
                    break;
                case 1:
                    lookAngle = maxLookAngle / 3;
                    break;
                case 2:
                    lookAngle = maxLookAngle / 2;
                    break;
                case 3:
                    lookAngle = maxLookAngle / 1;
                    break;
                case 4:
                    lookAngle = -maxLookAngle / 3;
                    break;
                case 5:
                    lookAngle = -maxLookAngle / 2;
                    break;
                case 6:
                    lookAngle = -maxLookAngle / 1;
                    break;
            }

            foreach (var item in raySensorsList)
            {
                item.transform.localRotation = Quaternion.Euler(0, lookAngle, 0);
            }
        }


        if (useAutoFocus)
        {
            var focus = act[1];
            switch (focus)
            {
                case 0:
                    focusAngle = 15;
                    break;
                case 1:
                    focusAngle = 30;
                    break;
                case 2:
                    focusAngle = 60;
                    break;
                case 3:
                    focusAngle = 90;
                    break;
                case 4:
                    focusAngle = 120;
                    break;
                case 5:
                    focusAngle = 180;
                    break;
            }
            //            var height = act[2];
            //            switch (height)
            //            {
            //                case 0:
            //                    rayHeight = 0;
            //                    break;
            //                case 1:
            //                    rayHeight = 1;
            //                    break;
            //            }

            foreach (var item in raySensorsList)
            {
                item.MaxRayDegrees = focusAngle;
                //                item.StartVerticalOffset = rayHeight;
                //                item.EndVerticalOffset = rayHeight;
            }
        }
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        // Move the agent using the action.
        //        MoveAgent(actionBuffers);
        MoveAgent(actionBuffers.DiscreteActions);

        // Penalty given each step to encourage agent to finish task quickly.
        AddReward(-1f / MaxStep);
    }

    //    /// <summary>
    //    /// Called every step of the engine. Here the agent takes an action.
    //    /// </summary>
    //    public override void OnActionReceived(ActionBuffers actionBuffers)
    //
    //    {
    //        // Move the agent using the action.
    //        MoveAgent(actionBuffers.DiscreteActions);
    //
    //        // Penalty given each step to encourage agent to finish task quickly.
    //        AddReward(-1f / MaxStep);
    //    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
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

    public void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("hazard"))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }

    public void SetBlockProperties()
    {
        var scale = m_ResetParams.GetWithDefault("block_scale", 1.5f);
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
}
