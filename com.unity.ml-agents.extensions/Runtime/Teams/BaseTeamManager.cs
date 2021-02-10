using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Teams
{
    public class BaseTeamManager : ITeamManager, IDisposable
    {
        int m_StepCount;
        int m_TeamMaxStep;
        readonly int m_Id = TeamManagerIdCounter.GetTeamManagerId();
        List<Agent> m_Agents = new List<Agent> { };


        public BaseTeamManager()
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
            if ((m_StepCount >= m_TeamMaxStep) && (m_TeamMaxStep > 0))
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
        /// Register the agent to the TeamManager.
        /// Registered agents will be able to receive team rewards from the TeamManager
        /// and share observations during training.
        /// </summary>
        public virtual void RegisterAgent(Agent agent)
        {
            if (!m_Agents.Contains(agent))
            {
                agent.SetTeamManager(this);
                m_Agents.Add(agent);
                agent.UnregisterFromTeamManager += UnregisterAgent;
            }
        }

        /// <summary>
        /// Remove the agent from the TeamManager.
        /// </summary>
        public virtual void UnregisterAgent(Agent agent)
        {
            if (m_Agents.Contains(agent))
            {
                m_Agents.Remove(agent);
                agent.UnregisterFromTeamManager -= UnregisterAgent;
            }
        }

        /// <summary>
        /// Get the ID of the TeamManager.
        /// </summary>
        /// <returns>
        /// TeamManager ID.
        /// </returns>
        public int GetId()
        {
            return m_Id;
        }

        /// <summary>
        /// Get list of all agents registered to this TeamManager.
        /// </summary>
        /// <returns>
        /// List of agents belongs to the TeamManager.
        /// </returns>
        public List<Agent> GetRegisteredAgents()
        {
            return m_Agents;
        }

        /// <summary>
        /// Add team reward for all agents under this Teammanager.
        /// Disabled agent will not receive this reward.
        /// </summary>
        public void AddTeamReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                if (agent.enabled)
                {
                    agent.AddTeamReward(reward);
                }
            }
        }

        /// <summary>
        /// Set team reward for all agents under this Teammanager.
        /// Disabled agent will not receive this reward.
        /// </summary>
        public void SetTeamReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                if (agent.enabled)
                {
                    agent.SetTeamReward(reward);
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

        public int TeamMaxStep
        {
            get { return m_TeamMaxStep; }
        }

        public void SetTeamMaxStep(int maxStep)
        {
            m_TeamMaxStep = maxStep;
        }

        /// <summary>
        /// End Episode for all agents under this TeamManager.
        /// </summary>
        public void EndTeamEpisode()
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

        /// <summary>
        /// End Episode for all agents under this TeamManager.
        /// </summary>
        public virtual void OnTeamEpisodeBegin()
        {

        }

        void Reset()
        {
            m_StepCount = 0;
            OnTeamEpisodeBegin();
        }
    }
}
