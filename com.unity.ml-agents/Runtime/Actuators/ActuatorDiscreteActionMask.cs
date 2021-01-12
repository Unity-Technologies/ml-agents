using System;
using System.Collections.Generic;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Implementation of IDiscreteActionMask that allows writing to the action mask from an <see cref="IActuator"/>.
    /// </summary>
    internal class ActuatorDiscreteActionMask : IDiscreteActionMask
    {
        /// When using discrete control, is the starting indices of the actions
        /// when all the branches are concatenated with each other.
        int[] m_StartingActionIndices;

        int[] m_BranchSizes;

        bool[] m_CurrentMask;

        IList<IActuator> m_Actuators;

        readonly int m_SumOfDiscreteBranchSizes;
        readonly int m_NumBranches;

        /// <summary>
        /// The offset into the branches array that is used when actuators are writing to the action mask.
        /// </summary>
        public int CurrentBranchOffset { get; set; }

        internal ActuatorDiscreteActionMask(IList<IActuator> actuators, int sumOfDiscreteBranchSizes, int numBranches, int[] branchSizes = null)
        {
            m_Actuators = actuators;
            m_SumOfDiscreteBranchSizes = sumOfDiscreteBranchSizes;
            m_NumBranches = numBranches;
            m_BranchSizes = branchSizes;
        }

        /// <inheritdoc/>
        public void WriteMask(int branch, IEnumerable<int> actionIndices)
        {
            LazyInitialize();

            // Perform the masking
            foreach (var actionIndex in actionIndices)
            {
#if DEBUG
                if (branch >= m_NumBranches || actionIndex >= m_BranchSizes[CurrentBranchOffset + branch])
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking: Action Mask is too large for specified branch.");
                }
#endif
                m_CurrentMask[actionIndex + m_StartingActionIndices[CurrentBranchOffset + branch]] = true;
            }
        }

        void LazyInitialize()
        {
            if (m_BranchSizes == null)
            {
                m_BranchSizes = new int[m_NumBranches];
                var start = 0;
                for (var i = 0; i < m_Actuators.Count; i++)
                {
                    var actuator = m_Actuators[i];
                    var branchSizes = actuator.ActionSpec.BranchSizes;
                    Array.Copy(branchSizes, 0, m_BranchSizes, start, branchSizes.Length);
                    start += branchSizes.Length;
                }
            }

            // By default, the masks are null. If we want to specify a new mask, we initialize
            // the actionMasks with trues.
            if (m_CurrentMask == null)
            {
                m_CurrentMask = new bool[m_SumOfDiscreteBranchSizes];
            }

            // If this is the first time the masked actions are used, we generate the starting
            // indices for each branch.
            if (m_StartingActionIndices == null)
            {
                m_StartingActionIndices = Utilities.CumSum(m_BranchSizes);
            }
        }

        /// <inheritdoc/>
        public bool[] GetMask()
        {
#if DEBUG
            if (m_CurrentMask != null)
            {
                AssertMask();
            }
#endif
            return m_CurrentMask;
        }

        /// <summary>
        /// Makes sure that the current mask is usable.
        /// </summary>
        void AssertMask()
        {
#if DEBUG
            for (var branchIndex = 0; branchIndex < m_NumBranches; branchIndex++)
            {
                if (AreAllActionsMasked(branchIndex))
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking : All the actions of branch " + branchIndex +
                        " are masked.");
                }
            }
#endif
        }

        /// <summary>
        /// Resets the current mask for an agent.
        /// </summary>
        public void ResetMask()
        {
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
    }
}
