using System.Collections.Generic;

namespace Unity.MLAgents.Extensions.Teams
{
    public class BaseTeamManager : ITeamManager
    {
        int m_StepCount;
        int m_TeamMaxStep;
        readonly int m_Id = TeamManagerIdCounter.GetTeamManagerId();
        List<Agent> m_Agents = new List<Agent> { };


        public BaseTeamManager()
        {
            Academy.Instance.TeamManagerStep += _ManagerStep;
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
        /// Registered agents will be able to receive team rewards from the TeamManager.
        /// All agents in the same training area should be added to the same TeamManager.
        /// </summary>
        public virtual void RegisterAgent(Agent agent)
        {
            if (!m_Agents.Contains(agent))
            {
                m_Agents.Add(agent);
            }
        }

        /// <summary>
        /// Remove the agent from the TeamManager.
        /// </summary>
        public virtual void RemoveAgent(Agent agent)
        {
            if (m_Agents.Contains(agent))
            {
                m_Agents.Remove(agent);
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
        public List<Agent> GetTeammates()
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
