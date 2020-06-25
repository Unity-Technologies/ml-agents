using System;
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
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/release_5_docs/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <param name="branch">The branch for which the actions will be masked.</param>
        /// <param name="actionIndices">The indices of the masked actions.</param>
        public void SetMask(int branch, IEnumerable<int> actionIndices)
        {
            m_Delegate.WriteMask(branch, actionIndices);
        }

        public void WriteMask(int branch, IEnumerable<int> actionIndices)
        {
            m_Delegate.WriteMask(branch, actionIndices);
        }

<<<<<<< HEAD
        public void WriteMask(int branch, IEnumerable<int> actionIndices)
||||||| constructed merge base
        /// <summary>
        /// Get the current mask for an agent.
        /// </summary>
        /// <returns>A mask for the agent. A boolean array of length equal to the total number of
        /// actions.</returns>
        public bool[] GetMask()
=======
        public bool[] GetMask()
>>>>>>> Get discrete action mask working and backward compatible.
        {
<<<<<<< HEAD
            m_Delegate.WriteMask(branch, actionIndices);
        }

        public bool[] GetMask()
        {
            return m_Delegate.GetMask();
||||||| constructed merge base
            if (m_CurrentMask != null)
            {
                AssertMask();
            }
            return m_CurrentMask;
        }

        /// <summary>
        /// Makes sure that the current mask is usable.
        /// </summary>
        void AssertMask()
        {
            // Action Masks can only be used in Discrete Control.
            if (m_BrainParameters.VectorActionSpaceType != SpaceType.Discrete)
            {
                throw new UnityAgentsException(
                    "Invalid Action Masking : Can only set action mask for Discrete Control.");
            }

            var numBranches = m_BrainParameters.VectorActionSize.Length;
            for (var branchIndex = 0; branchIndex < numBranches; branchIndex++)
            {
                if (AreAllActionsMasked(branchIndex))
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking : All the actions of branch " + branchIndex +
                        " are masked.");
                }
            }
=======
            return m_Delegate.GetMask();
>>>>>>> Get discrete action mask working and backward compatible.
        }

        public void ResetMask()
        {
<<<<<<< HEAD
            m_Delegate.ResetMask();
        }
||||||| constructed merge base
            if (m_CurrentMask != null)
            {
                Array.Clear(m_CurrentMask, 0, m_CurrentMask.Length);
            }
        }

        /// <summary>
        /// Checks if all the actions in the input branch are masked.
        /// </summary>
        /// <param name="branch"> The index of the branch to check.</param>
        /// <returns> True if all the actions of the branch are masked.</returns>
        bool AreAllActionsMasked(int branch)
        {
            if (m_CurrentMask == null)
            {
                return false;
            }
            var start = m_StartingActionIndices[branch];
            var end = m_StartingActionIndices[branch + 1];
            for (var i = start; i < end; i++)
            {
                if (!m_CurrentMask[i])
                {
                    return false;
                }
            }
            return true;
        }
=======
            m_Delegate.ResetMask();
        }

        public int CurrentBranchOffset => m_Delegate.CurrentBranchOffset;
>>>>>>> Get discrete action mask working and backward compatible.
    }
}
