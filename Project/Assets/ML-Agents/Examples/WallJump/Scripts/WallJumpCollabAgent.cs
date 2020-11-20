//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;

public class WallJumpCollabAgent : WallJumpAgent
{
    Vector3 m_InitialPosition;

    WallAreaScoring m_Scoring;
    public override void Initialize()
    {
        m_WallJumpSettings = FindObjectOfType<WallJumpSettings>();
        m_Scoring = ground.GetComponent<WallAreaScoring>();
        m_Configuration = 5;

        m_AgentRb = GetComponent<Rigidbody>();
        // m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        m_InitialPosition = transform.localPosition;
        spawnArea.SetActive(false);

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }
    public override void OnEpisodeBegin()
    {
        transform.localPosition = m_InitialPosition;
        m_Configuration = 5;
        m_AgentRb.velocity = default(Vector3);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);
        if (!Physics.Raycast(m_AgentRb.position, Vector3.down, 20))
        {
            m_Scoring.LoseCondition();
        }
    }

    protected override void ConfigureAgent(int config)
    {
        var localScale = wall.transform.localScale;
        var height = m_ResetParams.GetWithDefault("big_wall_height", 10);
        localScale = new Vector3(
            localScale.x,
            height,
            localScale.z);
        wall.transform.localScale = localScale;
    }

    // Detect when the agent hits the goal
    protected override void OnTriggerStay(Collider col)
    {
        if (col.gameObject.CompareTag("goal") && DoGroundCheck(true))
        {
            m_Scoring.WinCondition();
        }
    }
}
