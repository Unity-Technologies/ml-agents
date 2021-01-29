using System.Collections.Generic;
using Unity.MLAgents;
using System.Linq;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;

public class PushBlockTeamManager : BaseTeamManager
{
    Dictionary<Agent, bool> m_AgentDone = new Dictionary<Agent, bool> { };


    public override void RegisterAgent(Agent agent)
    {
        m_AgentDone[agent] = false;
    }

    public override void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
    {
        m_AgentDone[agent] = true;
    }

    public void OnTeamDone()
    {
        foreach (var agent in m_AgentDone.Keys.ToList())
        {
            if (m_AgentDone[agent])
            {
                agent.SendDoneToTrainer();
                m_AgentDone[agent] = false;
            }
        }
    }

    public void AddTeamReward(float reward)
    {
        foreach (var agent in m_AgentDone.Keys)
        {
            if (m_AgentDone[agent])
            {
                agent.AddRewardAfterDeath(reward);
            }
            else
            {
                agent.AddReward(reward);
            }
        }
    }
}
