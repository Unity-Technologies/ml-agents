using System.Collections.Generic;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents
{
    /// <summary>
    /// The DiscreteActionMasker class represents a set of masked (disallowed) actions and
    /// provides utilities for setting and retrieving them.
    /// </summary>
    /// <remarks>
    /// Agents that take discrete actions can explicitly indicate that specific actions
    /// are not allowed at a point in time. This enables the agent to indicate that some actions
    /// may be illegal. For example, if an agent is adjacent to a wall or other obstacle
    /// you could mask any actions that direct the agent to move into the blocked space.
    /// </remarks>
    public class DiscreteActionMasker : IDiscreteActionMask
    {
        IDiscreteActionMask m_Delegate;

        internal DiscreteActionMasker(IDiscreteActionMask actionMask)
        {
            m_Delegate = actionMask;
        }

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
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/release_12_docs/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <param name="branch">The branch for which the actions will be masked.</param>
        /// <param name="actionIndices">The indices of the masked actions.</param>
        public void SetMask(int branch, IEnumerable<int> actionIndices)
        {
            m_Delegate.WriteMask(branch, actionIndices);
        }

        /// <inheritdoc />
        public void WriteMask(int branch, IEnumerable<int> actionIndices)
        {
            m_Delegate.WriteMask(branch, actionIndices);
        }

        /// <inheritdoc />
        public bool[] GetMask()
        {
            return m_Delegate.GetMask();
        }

        /// <inheritdoc />
        public void ResetMask()
        {
            m_Delegate.ResetMask();
        }
    }
}
