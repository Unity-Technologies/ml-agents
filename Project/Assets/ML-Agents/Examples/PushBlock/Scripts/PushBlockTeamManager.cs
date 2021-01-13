using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;

public class PushBlockTeamManager : BaseTeamManager
{
    List<Agent> m_AgentList = new List<Agent> { };


    public override void RegisterAgent(Agent agent)
    {
        m_AgentList.Add(agent);
    }
}
