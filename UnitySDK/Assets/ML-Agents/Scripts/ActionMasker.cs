using System;
using System.Collections.Generic;
using System.Linq;

namespace MLAgents
{
    public class ActionMasker
    {
        /// When using discrete control, is the starting indices of the actions
        /// when all the branches are concatenated with each other.
        private int[] _startingActionIndices;

        private bool[] _currentMask;

        private readonly BrainParameters _brainParameters;

        public ActionMasker(BrainParameters brainParameters)
        {
            this._brainParameters = brainParameters;
        }
        
        /// <summary>
        /// Modifies an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the action passed as argument at the next decision. If no branch is
        /// specified, the default branch will be 0. The actionIndex or actionIndices correspond
        /// to the action the agent will be unable to perform.
        /// </summary>
        /// <param name="branch">The branch for which the actions will be masked</param>
        /// <param name="actionIndices">The indices of the masked actions</param>
        public void SetActionMask(int branch, IEnumerable<int> actionIndices)
        {   
            // If the branch does not exist, raise an error
            if (branch >= _brainParameters.vectorActionSize.Length )
                throw new UnityAgentsException(
                    "Invalid Action Masking : Branch "+branch+" does not exist.");

            int totalNumberActions = _brainParameters.vectorActionSize.Sum();
            
            // By default, the masks are null. If we want to specify a new mask, we initialize
            // the actionMasks with trues.
            if (_currentMask == null)
            {
                _currentMask = new bool[totalNumberActions];
            }

            // If this is the first time the masked actions are used, we generate the starting
            // indices for each branch.
            if (_startingActionIndices == null)
            {
                _startingActionIndices = Utilities.CumSum(_brainParameters.vectorActionSize);
            }
            
            // Perform the masking
            foreach (var actionIndex in actionIndices)
            {
                if (actionIndex >= _brainParameters.vectorActionSize[branch])
                {
                    throw new UnityAgentsException(
                        "Invalid Action Masking: Action Mask is too large for specified branch.");
                }
                _currentMask[actionIndex + _startingActionIndices[branch]] = true;
            } 
        }

        /// <summary>
        /// Get the current mask for an agent
        /// </summary>
        /// <returns>A mask for the agent. A boolean array of length equal to the total number of
        /// actions.</returns>
        public bool[] GetMask()
        {
            if (_currentMask != null)
            {
                AssertMask();
            }
            return _currentMask;
        }

        /// <summary>
        /// Makes sure that the current mask is usable.
        /// </summary>
        private void AssertMask()
        {
            // Action Masks can only be used in Discrete Control.
            if (_brainParameters.vectorActionSpaceType != SpaceType.discrete)
            {
                throw new UnityAgentsException(
                    "Invalid Action Masking : Can only set action mask for Discrete Control.");
            }
            
            var numBranches = _brainParameters.vectorActionSize.Length;
            for (var branchIndex = 0 ; branchIndex < numBranches; branchIndex++ )
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
        public void ResetMask()
        {
            if (_currentMask != null)
            {
                Array.Clear(_currentMask, 0, _currentMask.Length);
            }
        }

        /// <summary>
        /// Checks if all the actions in the input branch are masked
        /// </summary>
        /// <param name="branch"> The index of the branch to check</param>
        /// <returns> True if all the actions of the branch are masked</returns>
        private bool AreAllActionsMasked(int branch)
        {
            if (_currentMask == null)
            {
                return false;
            }
            var start = _startingActionIndices[branch];
            var end = _startingActionIndices[branch + 1];
            for (var i = start; i < end; i++)
            {
                if (!_currentMask[i])
                {
                    return false;
                }
            }
            return true;

        }
    }
}
