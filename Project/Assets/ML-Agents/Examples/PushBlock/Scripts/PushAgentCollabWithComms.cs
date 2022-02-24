//Put this script on your blue cube.
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class PushAgentCollabWithComms : Agent
{

    private PushBlockSettings m_PushBlockSettings;
    private Rigidbody m_AgentRb;  //cached on initialization

    protected override void Awake()
    {
        base.Awake();
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
    }

    public override void Initialize()
    {
        // Cache the agent rb
        m_AgentRb = GetComponent<Rigidbody>();
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    // public void MoveAgent(ActionSegment<int> act)
    public void MoveAgent(ActionBuffers act)
    {
        var continuousActions = act.ContinuousActions;
        var discreteActions = act.DiscreteActions;


        //ADD FORCE
        var inputV = continuousActions[0];
        var inputH = continuousActions[1];
        var rotate = continuousActions[2];

        var moveDir = transform.TransformDirection(new Vector3(inputH, 0, inputV));
        Quaternion rotAmount = Quaternion.AngleAxis(Time.fixedDeltaTime * 200f * rotate, Vector3.up);
        m_AgentRb.MoveRotation(m_AgentRb.rotation * rotAmount);
        m_AgentRb.AddForce(moveDir * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        // Move the agent using the action.
        MoveAgent(actionBuffers);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var contActs = actionsOut.ContinuousActions;
        contActs[0] = Input.GetAxisRaw("Vertical");
        contActs[1] = Input.GetAxisRaw("Horizontal");
        var rotLeft = Input.GetKey(KeyCode.J);
        var rotRight = Input.GetKey(KeyCode.K);
        var rotDir = rotLeft && rotRight ? 0 : rotLeft && !rotRight ? -1 : !rotLeft && rotRight ? 1 : 0;
        contActs[2] = rotDir;
    }
}
