using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

[System.Serializable]
public class PlayerState
{
    public int playerIndex;
    [FormerlySerializedAs("agentRB")]
    public Rigidbody agentRb;
    public Vector3 startingPos;
    public AgentSoccer agentScript;
    public float ballPosReward;
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
    [HideInInspector]
    public Vector3 ballStartingPos;
    public GameObject goalTextUI;
    [HideInInspector]
    public bool canResetBall;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;

    SoccerSettings m_SoccerSettings;

    public IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
    }

    void Awake()
    {
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        m_GroundRenderer = centerPitch.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        canResetBall = true;
        if (goalTextUI) { goalTextUI.SetActive(false); }
        ballRb = ball.GetComponent<Rigidbody>();
        m_BallController = ball.GetComponent<SoccerBallController>();
        m_BallController.area = this;
        ballStartingPos = ball.transform.position;
    }

    IEnumerator ShowGoalUI()
    {
        if (goalTextUI) goalTextUI.SetActive(true);
        yield return new WaitForSeconds(.25f);
        if (goalTextUI) goalTextUI.SetActive(false);
    }

    public void AllPlayersDone(float reward)
    {
        foreach (var ps in playerStates)
        {
            if (ps.agentScript.gameObject.activeInHierarchy)
            {
                if (reward != 0)
                {
                    ps.agentScript.AddReward(reward);
                }
                ps.agentScript.Done();
            }
        }
    }

    public void GoalTouched(AgentSoccer.Team scoredTeam)
    {
        foreach (var ps in playerStates)
        {
            if (ps.agentScript.team == scoredTeam)
            {
                RewardOrPunishPlayer(ps, m_SoccerSettings.strikerReward, m_SoccerSettings.goalieReward);
            }
            else
            {
                RewardOrPunishPlayer(ps, m_SoccerSettings.strikerPunish, m_SoccerSettings.goaliePunish);
            }
            if (m_SoccerSettings.randomizePlayersTeamForTraining)
            {
                ps.agentScript.ChooseRandomTeam();
            }

            if (scoredTeam == AgentSoccer.Team.Purple)
            {
                StartCoroutine(GoalScoredSwapGroundMaterial(m_SoccerSettings.purpleMaterial, 1));
            }
            else
            {
                StartCoroutine(GoalScoredSwapGroundMaterial(m_SoccerSettings.blueMaterial, 1));
            }
            if (goalTextUI)
            {
                StartCoroutine(ShowGoalUI());
            }
        }
    }

    public void RewardOrPunishPlayer(PlayerState ps, float striker, float goalie)
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
        var xOffset = 0f;
        if (role == AgentSoccer.AgentRole.Goalie)
        {
            xOffset = 13f;
        }
        if (role == AgentSoccer.AgentRole.Striker)
        {
            xOffset = 7f;
        }
        if (team == AgentSoccer.Team.Blue)
        {
            xOffset = xOffset * -1f;
        }
        var randomSpawnPos = ground.transform.position +
            new Vector3(xOffset, 0f, 0f)
            + (Random.insideUnitSphere * 2);
        randomSpawnPos.y = ground.transform.position.y + 2;
        return randomSpawnPos;
    }

    public Vector3 GetBallSpawnPosition()
    {
        var randomSpawnPos = ground.transform.position +
            new Vector3(0f, 0f, 0f)
            + (Random.insideUnitSphere * 2);
        randomSpawnPos.y = ground.transform.position.y + 2;
        return randomSpawnPos;
    }

    public void ResetBall()
    {
        ball.transform.position = GetBallSpawnPosition();
        ballRb.velocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;

        var ballScale = Academy.Instance.FloatProperties.GetPropertyWithDefault("ball_scale", 0.015f);
        ballRb.transform.localScale = new Vector3(ballScale, ballScale, ballScale);
    }
}
