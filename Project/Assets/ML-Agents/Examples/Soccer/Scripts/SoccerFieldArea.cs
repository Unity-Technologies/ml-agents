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

    void Awake()
    {
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

    public void GoalTouched(AgentSoccer.Team scoredTeam)
    {
        foreach (var ps in playerStates)
        {
            if (ps.agentScript.team == scoredTeam)
            {
                ps.agentScript.AddReward(1);
            }
            else
            {
                ps.agentScript.AddReward(-1);
            }
            ps.agentScript.EndEpisode();  //all agents need to be reset

            if (goalTextUI)
            {
                StartCoroutine(ShowGoalUI());
            }
        }
    }

    public Vector3 GetBallSpawnPosition()
    {
        var randomSpawnPos = ground.transform.position +
            new Vector3(0f, 0f, 0f);
        randomSpawnPos.y = ground.transform.position.y + .5f;
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
