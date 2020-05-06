using UnityEngine;
using Unity.MLAgentsExamples;

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
        smallAgents[0].Teammate1 = smallAgents[1];
        smallAgents[0].Teammate2 = smallAgents[2];
        smallAgents[0].TeammateRb1 = smallAgents[1].GetComponent<Rigidbody>();
        smallAgents[0].TeammateRb2 = smallAgents[2].GetComponent<Rigidbody>();
        smallAgents[1].Teammate1 = smallAgents[0];
        smallAgents[1].Teammate2 = smallAgents[2];
        smallAgents[1].TeammateRb1 = smallAgents[0].GetComponent<Rigidbody>();
        smallAgents[1].TeammateRb2 = smallAgents[2].GetComponent<Rigidbody>();
        smallAgents[2].Teammate1 = smallAgents[0];
        smallAgents[2].Teammate2 = smallAgents[1];
        smallAgents[2].TeammateRb1 = smallAgents[0].GetComponent<Rigidbody>();
        smallAgents[2].TeammateRb2 = smallAgents[1].GetComponent<Rigidbody>();

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
