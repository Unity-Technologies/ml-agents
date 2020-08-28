using System.Collections.Generic;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Interface for writing a mask to disable discrete actions for agents for the next decision.
    /// </summary>
    public interface IDiscreteActionMask
    {
        /// <summary>
        /// Modifies an action mask for discrete control agents.
        /// </summary>
        /// <remarks>
        /// When used, the agent will not be able to perform the actions passed as argument
        /// at the next decision for the specified action branch. The actionIndices correspond
        /// to the action options the agent will be unable to perform.
        ///
        /// See [Agents - Actions] for more information on masking actions.
        ///
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/release_2_docs/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <param name="branch">The branch for which the actions will be masked.</param>
        /// <param name="actionIndices">The indices of the masked actions.</param>
        void WriteMask(int branch, IEnumerable<int> actionIndices);

        /// <summary>
        /// Get the current mask for an agent.
        /// </summary>
        /// <returns>A mask for the agent. A boolean array of length equal to the total number of
        /// actions.</returns>
        bool[] GetMask();

        /// <summary>
        /// Resets the current mask for an agent.
        /// </summary>
        void ResetMask();
    }
}
