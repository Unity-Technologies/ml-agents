using System;
using System.Collections.Generic;
using System.Linq;
using MLAgents.Policies;

namespace MLAgents
{
    /// <summary>
    /// Agents that take discrete actions can explicitly indicate that specific actions
    /// are not allowed at a point in time. This enables the agent to indicate that some actions
    /// may be illegal (e.g. the King in Chess taking a move to the left if it is already in the
    /// left side of the board). This class represents the set of masked actions and provides
    /// the utilities for setting and retrieving them.
    /// </summary>
    public class DiscreteActionMasker
    {
        /// When using discrete control, is the starting indices of the actions
        /// when all the branches are concatenated with each other.
        int[] m_StartingActionIndices;

        bool[] m_CurrentMask;

        readonly BrainParameters m_BrainParameters;

        internal DiscreteActionMasker(BrainParameters brainParameters)
        {
            m_BrainParameters = brainParameters;
        }

        /// <summary>
        /// Modifies an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the actions passed as argument at the next decision for the specified
        /// action branch. The actionIndices correspond to the action options the agent will
        /// be unable to perform.
        /// </summary>
        /// <param name="branch">The branch for which the actions will be masked</param>
        /// <param name="actionIndices">The indices of the masked actions</param>
        public void SetMask(int branch, IEnumerable<int> actionIndices)
        {
            // If the branch does not exist, raise an error
            if (branch >= m_BrainParameters.vectorActionSize.Length)
                throw new UnityAgentsException(
                    "Invalid Action Masking : Branch " + branch + " does not exist.");

            var totalNumberActions = m_BrainParameters.vectorActionSize.Sum();

            // By default, the masks are null. If we want to specify a new mask, we initialize
            // the actionMasks with trues.
            if (m_CurrentMask == null)
            {
                m_CurrentMask = new bool[totalNumberActions];
            }

            // If this is the first time the masked actions are used, we generate the starting
            // indices for each branch.
            if (m_StartingActionIndices == null)
            {
                m_StartingActionIndices = Utilities.CumSum(m_BrainParameters.vectorActionSize);
            }

            // Perform the masking
            foreach (var actionIndex in actionIndices)
            {
                if (actionIndex >= m_BrainParameters.vectorActionSize[branch])
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking: Action Mask is too large for specified branch.");
                }
                m_CurrentMask[actionIndex + m_StartingActionIndices[branch]] = true;
            }
        }

        /// <summary>
        /// Get the current mask for an agent
        /// </summary>
        /// <returns>A mask for the agent. A boolean array of length equal to the total number of
        /// actions.</returns>
        internal bool[] GetMask()
        {
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
            if (m_BrainParameters.vectorActionSpaceType != SpaceType.Discrete)
            {
                throw new UnityAgentsException(
                    "Invalid Action Masking : Can only set action mask for Discrete Control.");
            }

            var numBranches = m_BrainParameters.vectorActionSize.Length;
            for (var branchIndex = 0; branchIndex < numBranches; branchIndex++)
            {
                if (AreAllActionsMasked(branchIndex))
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking : All the actions of branch " + branchIndex +
                        " are masked.");
                }
            }
        }

        /// <summary>
        /// Resets the current mask for an agent
        /// </summary>
        internal void ResetMask()
        {
            if (m_CurrentMask != null)
            {
                Array.Clear(m_CurrentMask, 0, m_CurrentMask.Length);
            }
        }

        /// <summary>
        /// Checks if all the actions in the input branch are masked
        /// </summary>
        /// <param name="branch"> The index of the branch to check</param>
        /// <returns> True if all the actions of the branch are masked</returns>
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
    }
}
