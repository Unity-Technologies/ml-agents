using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.MultiAgent
{
    public class BaseMultiAgentGroup : IMultiAgentGroup, IDisposable
    {
        int m_StepCount;
        int m_GroupMaxStep;
        readonly int m_Id = MultiAgentGroupIdCounter.GetGroupId();
        List<Agent> m_Agents = new List<Agent> { };


        public BaseMultiAgentGroup()
        {
            Academy.Instance.PostAgentAct += _ManagerStep;
        }

        public void Dispose()
        {
            Academy.Instance.PostAgentAct -= _ManagerStep;
            while (m_Agents.Count > 0)
            {
                UnregisterAgent(m_Agents[0]);
            }
        }

        void _ManagerStep()
        {
            m_StepCount += 1;
            if ((m_StepCount >= m_GroupMaxStep) && (m_GroupMaxStep > 0))
            {
                foreach (var agent in m_Agents)
                {
                    if (agent.enabled)
                    {
                        agent.EpisodeInterrupted();
                    }
                }
                Reset();
            }
        }

        /// <summary>
        /// Register the agent to the MultiAgentGroup.
        /// Registered agents will be able to receive group rewards from the MultiAgentGroup
        /// and share observations during training.
        /// </summary>
        public virtual void RegisterAgent(Agent agent)
        {
            if (!m_Agents.Contains(agent))
            {
                agent.SetMultiAgentGroup(this);
                m_Agents.Add(agent);
                agent.UnregisterFromGroup += UnregisterAgent;
            }
        }

        /// <summary>
        /// Remove the agent from the MultiAgentGroup.
        /// </summary>
        public virtual void UnregisterAgent(Agent agent)
        {
            if (m_Agents.Contains(agent))
            {
                m_Agents.Remove(agent);
                agent.UnregisterFromGroup -= UnregisterAgent;
            }
        }

        /// <summary>
        /// Get the ID of the MultiAgentGroup.
        /// </summary>
        /// <returns>
        /// MultiAgentGroup ID.
        /// </returns>
        public int GetId()
        {
            return m_Id;
        }

        /// <summary>
        /// Get list of all agents registered to this MultiAgentGroup.
        /// </summary>
        /// <returns>
        /// List of agents belongs to the MultiAgentGroup.
        /// </returns>
        public List<Agent> GetRegisteredAgents()
        {
            return m_Agents;
        }

        /// <summary>
        /// Add group reward for all agents under this MultiAgentGroup.
        /// Disabled agent will not receive this reward.
        /// </summary>
        public void AddGroupReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                if (agent.enabled)
                {
                    agent.AddGroupReward(reward);
                }
            }
        }

        /// <summary>
        /// Set group reward for all agents under this MultiAgentGroup.
        /// Disabled agent will not receive this reward.
        /// </summary>
        public void SetGroupReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                if (agent.enabled)
                {
                    agent.SetGroupReward(reward);
                }
            }
        }

        /// <summary>
        /// Returns the current step counter (within the current episode).
        /// </summary>
        /// <returns>
        /// Current step count.
        /// </returns>
        public int StepCount
        {
            get { return m_StepCount; }
        }

        public int GroupMaxStep
        {
            get { return m_GroupMaxStep; }
        }

        public void SetGroupMaxStep(int maxStep)
        {
            m_GroupMaxStep = maxStep;
        }

        /// <summary>
        /// End Episode for all agents under this MultiAgentGroup.
        /// </summary>
        public void EndGroupEpisode()
        {
            foreach (var agent in m_Agents)
            {
                if (agent.enabled)
                {
                    agent.EndEpisode();
                }
            }
            Reset();
        }

        void Reset()
        {
            m_StepCount = 0;
        }
    }
}
