using System;
using System.Linq;
using System.Collections.Generic;

namespace Unity.MLAgents
{
    /// <summary>
    /// A basic class implementation of MultiAgentGroup.
    /// </summary>
    public class SimpleMultiAgentGroup : IMultiAgentGroup, IDisposable
    {
        readonly int m_Id = MultiAgentGroupIdCounter.GetGroupId();
        HashSet<Agent> m_Agents = new HashSet<Agent>();

        /// <summary>
        /// Disposes of the SimpleMultiAgentGroup.
        /// </summary>
        public virtual void Dispose()
        {
            while (m_Agents.Count > 0)
            {
                UnregisterAgent(m_Agents.First());
            }
        }

        /// <inheritdoc />
        public virtual void RegisterAgent(Agent agent)
        {
            if (!m_Agents.Contains(agent))
            {
                agent.SetMultiAgentGroup(this);
                m_Agents.Add(agent);
                agent.OnAgentDisabled += UnregisterAgent;
            }
        }

        /// <inheritdoc />
        public virtual void UnregisterAgent(Agent agent)
        {
            if (m_Agents.Contains(agent))
            {
                agent.SetMultiAgentGroup(null);
                m_Agents.Remove(agent);
                agent.OnAgentDisabled -= UnregisterAgent;
            }
        }

        /// <inheritdoc />
        public int GetId()
        {
            return m_Id;
        }

        /// <summary>
        /// Get list of all agents currently registered to this MultiAgentGroup.
        /// </summary>
        /// <returns>
        /// List of agents registered to the MultiAgentGroup.
        /// </returns>
        public IReadOnlyCollection<Agent> GetRegisteredAgents()
        {
            return m_Agents;
        }

        /// <summary>
        /// Increments the group rewards for all agents in this MultiAgentGroup.
        /// </summary>
        /// <remarks>
        /// This function increases or decreases the group rewards by a given amount for all agents
        /// in the group. Use <see cref="SetGroupReward(float)"/> to set the group reward assigned
        /// to the current step with a specific value rather than increasing or decreasing it.
        ///
        /// A positive group reward indicates the whole group's accomplishments or desired behaviors.
        /// Every agent in the group will receive the same group reward no matter whether the
        /// agent's act directly leads to the reward. Group rewards are meant to reinforce agents
        /// to act in the group's best interest instead of individual ones.
        /// Group rewards are treated differently than individual agent rewards during training, so
        /// calling AddGroupReward() is not equivalent to calling agent.AddReward() on each agent in the group.
        /// </remarks>
        /// <param name="reward">Incremental group reward value.</param>
        public void AddGroupReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                agent.AddGroupReward(reward);
            }
        }

        /// <summary>
        /// Set the group rewards for all agents in this MultiAgentGroup.
        /// </summary>
        /// <remarks>
        /// This function replaces any group rewards given during the current step for all agents in the group.
        /// Use <see cref="AddGroupReward(float)"/> to incrementally change the group reward rather than
        /// overriding it.
        ///
        /// A positive group reward indicates the whole group's accomplishments or desired behaviors.
        /// Every agent in the group will receive the same group reward no matter whether the
        /// agent's act directly leads to the reward. Group rewards are meant to reinforce agents
        /// to act in the group's best interest instead of indivisual ones.
        /// Group rewards are treated differently than individual agent rewards during training, so
        /// calling SetGroupReward() is not equivalent to calling agent.SetReward() on each agent in the group.
        /// </remarks>
        /// <param name="reward">The new value of the group reward.</param>
        public void SetGroupReward(float reward)
        {
            foreach (var agent in m_Agents)
            {
                agent.SetGroupReward(reward);
            }
        }

        /// <summary>
        /// End episodes for all agents in this MultiAgentGroup.
        /// </summary>
        /// <remarks>
        /// This should be used when the episode can no longer continue, such as when the group
        /// reaches the goal or fails at the task.
        /// </remarks>
        public void EndGroupEpisode()
        {
            foreach (var agent in m_Agents)
            {
                agent.EndEpisode();
            }
        }

        /// <summary>
        /// Indicate that the episode is over but not due to the "fault" of the group.
        /// This has the same end result as calling <see cref="EndGroupEpisode"/>, but has a
        /// slightly different effect on training.
        /// </summary>
        /// <remarks>
        /// This should be used when the episode could continue, but has gone on for
        /// a sufficient number of steps, such as if the environment hits some maximum number of steps.
        /// </remarks>
        public void GroupEpisodeInterrupted()
        {
            foreach (var agent in m_Agents)
            {
                agent.EpisodeInterrupted();
            }
        }
    }
}
