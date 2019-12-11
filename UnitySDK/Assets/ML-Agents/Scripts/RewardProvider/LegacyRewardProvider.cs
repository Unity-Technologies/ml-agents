namespace MLAgents.RewardProvider
{
    /// <summary>
    /// A legacy reward provider that can be used in an Agent as a way to easily upgrade
    /// from the old reward system.
    /// </summary>
    public class LegacyRewardProvider : IRewardProvider
    {
        float m_IncrementalReward;
        float m_CumulativeReward;
        
        public float GetIncrementalReward()
        {
            return m_IncrementalReward;
        }
        
        
        /// <summary>
        /// Resets the step reward and possibly the episode reward for the agent.
        /// </summary>
        public void ResetReward(bool done = false)
        {
            m_IncrementalReward = 0f;
            if (done)
            {
                m_CumulativeReward = 0f;
            }
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
            m_CumulativeReward += (reward - m_IncrementalReward);
            m_IncrementalReward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
            m_IncrementalReward += increment;
            m_CumulativeReward += increment;
        }

        /// <summary>
        /// Retrieves the step reward for the Agent.
        /// </summary>
        /// <returns>The step reward.</returns>
        public float GetReward()
        {
            return m_IncrementalReward;
        }

        /// <summary>
        /// Retrieves the episode reward for the Agent.
        /// </summary>
        /// <returns>The episode reward.</returns>
        public float GetCumulativeReward()
        {
            return m_CumulativeReward;
        }
    }
}
