using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class HallwayTeamManager : BaseTeamManager
{
    List<Agent> m_AgentList = new List<Agent> { };


    public override void RegisterAgent(Agent agent)
    {
        m_AgentList.Add(agent);
    }

    public override void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
    {
        // Possible implementation - save reference to Agent's IPolicy so that we can repeatedly
        // call IPolicy.RequestDecision on behalf of the Agent after it's dead
        // If so, we'll need dummy sensor impls with the same shape as the originals.
        agent.SendDoneToTrainer();
    }

    public override void AddTeamReward(float reward)
    {

    }
}
