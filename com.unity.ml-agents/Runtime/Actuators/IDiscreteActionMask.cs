namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Interface for writing a mask to disable discrete actions for agents for the next decision.
    /// </summary>
    public interface IDiscreteActionMask
    {
        /// <summary>
        /// Set whether or not the action index for the given branch is allowed.
        /// </summary>
        /// <remarks>
        /// By default, all discrete actions are allowed.
        /// If isEnabled is false, the agent will not be able to perform the actions passed as argument
        /// at the next decision for the specified action branch. The actionIndex corresponds
        /// to the action options the agent will be unable to perform.
        ///
        /// See [Agents - Actions] for more information on masking actions.
        ///
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Learning-Environment-Design-Agents.md#masking-discrete-actions
        /// </remarks>
        /// <param name="branch">The branch for which the actions will be masked.</param>
        /// <param name="actionIndex">Index of the action.</param>
        /// <param name="isEnabled">Whether the action is allowed or not.</param>
        void SetActionEnabled(int branch, int actionIndex, bool isEnabled);
    }
}
