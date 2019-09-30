using UnityEngine;
using MLAgents;

public class AgentSoccer : Agent
{
    public enum Team
    {
        Purple,
        Blue
    }
    public enum AgentRole
    {
        Striker,
        Goalie
    }

    public Team team;
    public AgentRole agentRole;
    float m_KickPower;
    int m_PlayerIndex;
    public SoccerFieldArea area;

    [HideInInspector]
    public Rigidbody agentRb;
    SoccerAcademy m_Academy;
    Renderer m_AgentRenderer;
    RayPerception m_RayPer;

    float[] m_RayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
    string[] m_DetectableObjectsPurple = { "ball", "purpleGoal", "blueGoal",
                                           "wall", "purpleAgent", "blueAgent" };
    string[] m_DetectableObjectsBlue = { "ball", "blueGoal", "purpleGoal",
                                         "wall", "blueAgent", "purpleAgent" };

    public void ChooseRandomTeam()
    {
        team = (Team)Random.Range(0, 2);
        if (team == Team.Purple)
        {
            JoinPurpleTeam(agentRole);
        }
        else
        {
            JoinBlueTeam(agentRole);
        }
    }

    public void JoinPurpleTeam(AgentRole role)
    {
        agentRole = role;
        team = Team.Purple;
        m_AgentRenderer.material = m_Academy.purpleMaterial;
        tag = "purpleAgent";
    }

    public void JoinBlueTeam(AgentRole role)
    {
        agentRole = role;
        team = Team.Blue;
        m_AgentRenderer.material = m_Academy.blueMaterial;
        tag = "blueAgent";
    }

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        m_AgentRenderer = GetComponentInChildren<Renderer>();
        m_RayPer = GetComponent<RayPerception>();
        m_Academy = FindObjectOfType<SoccerAcademy>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        var playerState = new PlayerState
        {
            agentRb = agentRb,
            startingPos = transform.position,
            agentScript = this,
        };
        area.playerStates.Add(playerState);
        m_PlayerIndex = area.playerStates.IndexOf(playerState);
        playerState.playerIndex = m_PlayerIndex;
    }

    public override void CollectObservations()
    {
        var rayDistance = 20f;
        string[] detectableObjects;
        if (team == Team.Purple)
        {
            detectableObjects = m_DetectableObjectsPurple;
        }
        else
        {
            detectableObjects = m_DetectableObjectsBlue;
        }
        AddVectorObs(m_RayPer.Perceive(rayDistance, m_RayAngles, detectableObjects, 0f, 0f));
        AddVectorObs(m_RayPer.Perceive(rayDistance, m_RayAngles, detectableObjects, 1f, 0f));
    }

    public void MoveAgent(float[] act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = Mathf.FloorToInt(act[0]);

        // Goalies and Strikers have slightly different action spaces.
        if (agentRole == AgentRole.Goalie)
        {
            m_KickPower = 0f;
            switch (action)
            {
                case 1:
                    dirToGo = transform.forward * 1f;
                    m_KickPower = 1f;
                    break;
                case 2:
                    dirToGo = transform.forward * -1f;
                    break;
                case 4:
                    dirToGo = transform.right * -1f;
                    break;
                case 3:
                    dirToGo = transform.right * 1f;
                    break;
            }
        }
        else
        {
            m_KickPower = 0f;
            switch (action)
            {
                case 1:
                    dirToGo = transform.forward * 1f;
                    m_KickPower = 1f;
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
        }
        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo * m_Academy.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        // Existential penalty for strikers.
        if (agentRole == AgentRole.Striker)
        {
            AddReward(-1f / 3000f);
        }
        // Existential bonus for goalies.
        if (agentRole == AgentRole.Goalie)
        {
            AddReward(1f / 3000f);
        }
        MoveAgent(vectorAction);
    }

    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = 2000f * m_KickPower;
        if (c.gameObject.CompareTag("ball"))
        {
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }

    public override void AgentReset()
    {
        if (m_Academy.randomizePlayersTeamForTraining)
        {
            ChooseRandomTeam();
        }

        if (team == Team.Purple)
        {
            JoinPurpleTeam(agentRole);
            transform.rotation = Quaternion.Euler(0f, -90f, 0f);
        }
        else
        {
            JoinBlueTeam(agentRole);
            transform.rotation = Quaternion.Euler(0f, 90f, 0f);
        }
        transform.position = area.GetRandomSpawnPos(agentRole, team);
        agentRb.velocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;
        SetResetParameters();
    }

    public void SetResetParameters()
    {
        area.ResetBall();
    }
}
