using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using Random = UnityEngine.Random;

[Serializable]
public class PlayerState
{
    public AgentSoccer agentScript;
}

public class SoccerFieldArea : MonoBehaviour
{
    public GameObject ball;
    [FormerlySerializedAs("ballRB")]
    [HideInInspector]
    public Rigidbody ballRb;
    public GameObject ground;
    public GameObject centerPitch;
    SoccerBallController m_BallController;
    public List<PlayerState> playerStates = new List<PlayerState>();
    Vector3 m_SpawnAreaSize;
    public GameObject goalTextUI;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    SoccerAcademy m_Academy;

    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
    }

    void Awake()
    {
        m_Academy = FindObjectOfType<SoccerAcademy>();
        m_GroundRenderer = centerPitch.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        if (goalTextUI) { goalTextUI.SetActive(false); }
        ballRb = ball.GetComponent<Rigidbody>();
        m_BallController = ball.GetComponent<SoccerBallController>();
        m_BallController.area = this;
    }

    IEnumerator ShowGoalUI()
    {
        if (goalTextUI) goalTextUI.SetActive(true);
        yield return new WaitForSeconds(.25f);
        if (goalTextUI) goalTextUI.SetActive(false);
    }

    public void GoalTouched(AgentSoccer.Team scoredTeam)
    {
        foreach (PlayerState ps in playerStates)
        {
            if (ps.agentScript.team == scoredTeam)
            {
                RewardOrPunishPlayer(ps, m_Academy.strikerReward, m_Academy.goalieReward);
            }
            else
            {
                RewardOrPunishPlayer(ps, m_Academy.strikerPunish, m_Academy.goaliePunish);
            }
            if (m_Academy.randomizePlayersTeamForTraining)
            {
                ps.agentScript.ChooseRandomTeam();
            }

            StartCoroutine(scoredTeam == AgentSoccer.Team.Red
                ? GoalScoredSwapGroundMaterial(m_Academy.redMaterial, 1)
                : GoalScoredSwapGroundMaterial(m_Academy.blueMaterial, 1));
            if (goalTextUI)
            {
                StartCoroutine(ShowGoalUI());
            }
        }
    }

    static void RewardOrPunishPlayer(PlayerState ps, float striker, float goalie)
    {
        if (ps.agentScript.agentRole == AgentSoccer.AgentRole.Striker)
        {
            ps.agentScript.AddReward(striker);
        }
        if (ps.agentScript.agentRole == AgentSoccer.AgentRole.Goalie)
        {
            ps.agentScript.AddReward(goalie);
        }
        ps.agentScript.Done();  //all agents need to be reset
    }

    public Vector3 GetRandomSpawnPos(AgentSoccer.AgentRole role, AgentSoccer.Team team)
    {
        float xOffset;
        switch (role)
        {
            case AgentSoccer.AgentRole.Goalie:
                xOffset = 13f;
                break;
            case AgentSoccer.AgentRole.Striker:
                xOffset = 7f;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(role), role, null);
        }

        if (team == AgentSoccer.Team.Blue)
        {
            xOffset *= -1f;
        }

        var position = ground.transform.position;
        var randomSpawnPos = position +
            new Vector3(xOffset, 0f, 0f)
            + Random.insideUnitSphere * 2;
        randomSpawnPos.y = position.y + 2;
        return randomSpawnPos;
    }

    Vector3 GetBallSpawnPosition()
    {
        var position = ground.transform.position;
        var randomSpawnPos = position +
            new Vector3(0f, 0f, 0f)
            + Random.insideUnitSphere * 2;
        randomSpawnPos.y = position.y + 2;
        return randomSpawnPos;
    }

    public void ResetBall()
    {
        ball.transform.position = GetBallSpawnPosition();
        ballRb.velocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;

        var ballScale = m_Academy.resetParameters["ball_scale"];
        ballRb.transform.localScale = new Vector3(ballScale, ballScale, ballScale);
    }
}
