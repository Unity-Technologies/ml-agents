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
        /// This function is called on every step of the simulation and should be
        /// used as a place to store an <see cref="Agent"/>'s incremental reward
        /// before the reward is sent off to the brain from the
        /// <see cref="GetIncrementalReward"/> method.
        /// </summary>
        void RewardStep();
    }
}
