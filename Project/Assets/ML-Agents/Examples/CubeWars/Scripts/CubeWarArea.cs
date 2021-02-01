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
            foreach (var smallAgent in smallAgents)
            {
                smallAgent.SetReward(-1.0f);
                smallAgent.EndEpisode();
            }
            foreach (var largeAgent in largeAgents)
            {
                largeAgent.SetReward(1.0f);
                largeAgent.EndEpisode();
            }

        }
        else if (!largeAlive)
        {
            foreach (var smallAgent in smallAgents)
            {
                smallAgent.SetReward(1.0f);
                smallAgent.EndEpisode();
            }
            foreach (var largeAgent in largeAgents)
            {
                largeAgent.SetReward(-1.0f);
                largeAgent.EndEpisode();
            }
        }
    }
}
