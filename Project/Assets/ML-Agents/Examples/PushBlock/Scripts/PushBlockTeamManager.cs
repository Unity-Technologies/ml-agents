using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;

public class PushBlockTeamManager : BaseTeamManager
{
    Dictionary<Agent, bool> m_AgentDoneState = new Dictionary<Agent, bool> { };


    public override void RegisterAgent(Agent agent)
    {
        m_AgentDoneState[agent] = false;
    }

    public override void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
    {
        m_AgentDoneState[agent] = true;
        if (!m_AgentDoneState.ContainsValue(false))
        {
            foreach (var doneAgent in m_AgentDoneState.Keys)
            {
                doneAgent.SendDoneToTrainer();
                m_AgentDoneState[doneAgent] = false;
            }
        }
    }

    public override void AddTeamReward(float reward)
    {
        foreach (var agent in m_AgentDoneState.Keys)
        {
            agent.AddReward(reward);
        }
    }
}
