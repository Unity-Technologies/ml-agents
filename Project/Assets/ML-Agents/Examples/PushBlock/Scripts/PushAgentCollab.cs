//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class PushAgentCollab : Agent
{

    PushBlockSettings m_PushBlockSettings;

    //    Rigidbody m_BlockRb;  //cached on initialization
    Rigidbody m_AgentRb;  //cached on initialization

    public bool useVectorObs = true;
    void Awake()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
    }

    public override void Initialize()
    {
        // Cache the agent rigidbody
        m_AgentRb = GetComponent<Rigidbody>();
    }

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
    }

    // public override void CollectObservations(VectorSensor sensor)
    // {
    //     if (useVectorObs)
    //     {
    //         sensor.AddObservation(StepCount / (float)MaxStep);
    //     }
    // }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        // Move the agent using the action.
        MoveAgent(actionBuffers.DiscreteActions);

        // Penalty given each step to encourage agent to finish task quickly.
        AddReward(-1f / MaxStep);
    }

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
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {
    }

}
