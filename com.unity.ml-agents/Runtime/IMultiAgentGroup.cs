namespace Unity.MLAgents
{
    /// <summary>
    /// MultiAgentGroup interface for grouping agents to support multi-agent training.
    /// </summary>
    public interface IMultiAgentGroup
    {
        /// <summary>
        /// Get the ID of MultiAgentGroup.
        /// </summary>
        /// <returns>
        /// MultiAgentGroup ID.
        /// </returns>
        int GetId();

        /// <summary>
        /// Register agent to the MultiAgentGroup.
        /// </summary>
        /// <param name="agent">The Agent to register.</param>
        void RegisterAgent(Agent agent);

        /// <summary>
        /// Unregister agent from the MultiAgentGroup.
        /// </summary>
        /// <param name="agent">The Agent to unregister.</param>
        void UnregisterAgent(Agent agent);
    }
}
