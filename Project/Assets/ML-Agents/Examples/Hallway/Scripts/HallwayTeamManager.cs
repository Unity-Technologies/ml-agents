using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;

public class HallwayTeamManager : BaseTeamManager
{
    List<Agent> m_AgentList = new List<Agent> { };

    public override void RegisterAgent(Agent agent)
    {
        m_AgentList.Add(agent);
    }

    // public override void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
    // {

    // }

    // public override void AddTeamReward(float reward)
    // {

    // }
}
