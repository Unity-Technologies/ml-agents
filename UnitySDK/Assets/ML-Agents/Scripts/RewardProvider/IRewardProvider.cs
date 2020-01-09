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
    }
}
