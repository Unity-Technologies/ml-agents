using UnityEngine;
using Unity.MLAgentsExamples;

public class CubeWarArea : Area
{
    [HideInInspector]
    public SmallCubeAgent[] smallAgents;
    [HideInInspector]
    public LargeCubeAgent[] largeAgents;
    public float range;
    private CubeWarsTeamManager m_SmallTeamManager;
    private CubeWarsTeamManager m_LargeTeamManager;
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 10000;
    private int m_ResetTimer = 0;


    void Start()
    {
        m_SmallTeamManager = new CubeWarsTeamManager();
        m_LargeTeamManager = new CubeWarsTeamManager();
        range = 1.0f;
        smallAgents = GetComponentsInChildren<SmallCubeAgent>();
        largeAgents = GetComponentsInChildren<LargeCubeAgent>();
        foreach (var agent in smallAgents)
        {
            agent.SetTeamManager(m_SmallTeamManager);
        }
        foreach (var agent in largeAgents)
        {
            agent.SetTeamManager(m_LargeTeamManager);
        }
    }


    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer > MaxEnvironmentSteps)
        {
            ResetAllAgents(true);
            m_ResetTimer = 0;
        }
    }

    public void ResetAllAgents(bool terminated = false)
    {
        foreach (var smallAgent in smallAgents)
        {
            if (terminated)
            {
                smallAgent.EpisodeInterrupted();
            }
            else
            {
                smallAgent.EndEpisode();
            }
            smallAgent.gameObject.SetActive(true);
        }
        foreach (var largeAgent in largeAgents)
        {
            if (terminated)
            {
                largeAgent.EpisodeInterrupted();
            }
            else
            {
                largeAgent.EndEpisode();
            }
        }
    }
    public void AgentDied()
    {
        bool smallAlive = false;
        foreach (var smallAgent in smallAgents)
        {
            if (!smallAgent.IsDead())
            {
                smallAlive = true;
            }
        }
        bool largeAlive = false;
        foreach (var largeAgent in largeAgents)
        {
            if (!largeAgent.IsDead())
            {
                largeAlive = true;
            }
        }
        if (!smallAlive)
        {
            Debug.Log("Big Agent Wins");
            foreach (var smallAgent in smallAgents)
            {
                smallAgent.SetReward(-1.0f);
            }
            foreach (var largeAgent in largeAgents)
            {
                largeAgent.SetReward(1.0f);
            }
            ResetAllAgents();

        }
        else if (!largeAlive)
        {
            Debug.Log("Small Agents Win");
            foreach (var smallAgent in smallAgents)
            {
                smallAgent.SetReward(1.0f);
            }
            foreach (var largeAgent in largeAgents)
            {
                largeAgent.SetReward(-1.0f);
            }
            ResetAllAgents();
        }
    }
}
