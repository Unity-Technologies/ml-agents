namespace MLAgents.RewardProvider
{
    /// <summary>
    /// Reward providers allow users to provide rewards for Agent behavior during training in order to
    /// give hints on what types of actions are "better" than others based on an Agent's previous observation.
    /// </summary>
    public interface IRewardProvider
    {
        /// <summary>
        /// Get an incremental reward to pass along to a trainer.  
        /// </summary>
        /// <returns></returns>
        float GetIncrementalReward();

        /// <summary>
        /// Adds a scalar value to the current reward for this step.
        /// </summary>
        void AddReward(float reward);

        void SetReward(float reward);

        /// <summary>
        /// Retrieves the step reward for the Agent.
        /// </summary>
        /// <returns>The step reward.</returns>
        float GetReward();
        
        /// <summary>
        /// Retrieves the episode reward for the Agent.
        /// </summary>
        /// <returns>The episode reward.</returns>
        float GetCumulativeReward();

        void ResetReward(bool done);
    }
}
