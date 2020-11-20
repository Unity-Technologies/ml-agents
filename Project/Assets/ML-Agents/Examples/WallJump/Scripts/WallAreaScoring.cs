using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAreaScoring : MonoBehaviour
{
    public GameObject[] agents;
    WallJumpSettings m_WallJumpSettings;
    Renderer m_GroundRenderer;
    Material m_GroundMaterial;

    protected IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); //wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void Start()
    {
        m_WallJumpSettings = FindObjectOfType<WallJumpSettings>();
        m_GroundRenderer = GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
    }

    public void WinCondition()
    {
        foreach (var agent in agents)
        {
            WallJumpCollabAgent agentScript = agent.GetComponent<WallJumpCollabAgent>();
            agentScript.SetReward(1f);
            agentScript.EndEpisode();
        }
        StartCoroutine(
            GoalScoredSwapGroundMaterial(m_WallJumpSettings.goalScoredMaterial, 1f));
    }

    public void LoseCondition()
    {
        foreach (var agent in agents)
        {
            WallJumpCollabAgent agentScript = agent.GetComponent<WallJumpCollabAgent>();
            agentScript.SetReward(-1f);
            agentScript.EndEpisode();

        }
        StartCoroutine(
            GoalScoredSwapGroundMaterial(m_WallJumpSettings.failMaterial, .2f));
    }
}
