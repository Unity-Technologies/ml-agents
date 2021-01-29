using System.Collections.Generic;
using Unity.MLAgents;
using System.Linq;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;
using System;

public class PushBlockTeamManager : BaseTeamManager
{
    float m_RewardDiscount = 1;

    // -1 means not done
    Dictionary<Agent, int> m_AgentDoneStep = new Dictionary<Agent, int> { };


    public override void RegisterAgent(Agent agent)
    {
        m_AgentDoneStep[agent] = -1;
    }

    public override void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
    {
        m_AgentDoneStep[agent] = agent.StepCount;
    }

    public void OnTeamDone()
    {
        foreach (var agent in m_AgentDoneStep.Keys.ToList())
        {
            if (m_AgentDoneStep[agent] >= 0)
            {
                agent.SendDoneToTrainer();
                m_AgentDoneStep[agent] = -1;
            }
        }
    }

    public void AddTeamReward(float reward)
    {
        int maxAgentStep = m_AgentDoneStep.Values.Max();
        foreach (var agent in m_AgentDoneStep.Keys)
        {
            if (m_AgentDoneStep[agent] >= 0)
            {
                agent.AddRewardAfterDeath(reward * (float)Math.Pow(m_RewardDiscount, maxAgentStep - m_AgentDoneStep[agent]));
            }
            else
            {
                agent.AddReward(reward);
            }
        }
    }

    public void SetRewardDiscount(float discountRate)
    {
        m_RewardDiscount = discountRate;
    }
}
