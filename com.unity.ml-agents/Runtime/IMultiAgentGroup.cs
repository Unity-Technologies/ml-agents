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
        void RegisterAgent(Agent agent);

        /// <summary>
        /// UnRegister agent from the MultiAgentGroup.
        /// </summary>
        void UnregisterAgent(Agent agent);
    }
}
