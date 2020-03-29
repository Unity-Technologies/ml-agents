using UnityEngine;
using MLAgentsExamples;

public class CubeWarArea : Area
{
    [HideInInspector]
    public SmallCubeAgent[] smallAgents;
    [HideInInspector]
    public LargeCubeAgent[] largeAgents;
    public float range;


    void Start()
    {
        range = 1.0f;
        smallAgents = GetComponentsInChildren<SmallCubeAgent>();
        largeAgents = GetComponentsInChildren<LargeCubeAgent>();
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
