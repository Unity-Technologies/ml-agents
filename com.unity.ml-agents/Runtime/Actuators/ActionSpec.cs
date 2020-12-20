using System;
using System.Linq;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Defines the structure of an Action Space to be used by the Actuator system.
    /// </summary>
    [Serializable]
    public struct ActionSpec
    {
        [SerializeField]
        int m_NumContinuousActions;

        /// <summary>
        /// An array of branch sizes for our action space.
        ///
        /// For an IActuator that uses a Discrete <see cref="SpaceType"/>, the number of
        /// branches is the Length of the Array and each index contains the branch size.
        /// The cumulative sum of the total number of discrete actions can be retrieved
        /// by the <see cref="SumOfDiscreteBranchSizes"/> property.
        ///
        /// For an IActuator with a Continuous it will be null.
        /// </summary>
        public int[] BranchSizes;

        /// <summary>
        /// The number of actions for a Continuous <see cref="SpaceType"/>.
        /// </summary>
        public int NumContinuousActions { get { return m_NumContinuousActions; } set { m_NumContinuousActions = value; } }

        /// <summary>
        /// The number of branches for a Discrete <see cref="SpaceType"/>.
        /// </summary>
        public int NumDiscreteActions { get { return BranchSizes == null ? 0 : BranchSizes.Length; } }

        /// <summary>
        /// Get the total number of Discrete Actions that can be taken by calculating the Sum
        /// of all of the Discrete Action branch sizes.
        /// </summary>
        public int SumOfDiscreteBranchSizes { get { return BranchSizes == null ? 0 : BranchSizes.Sum(); } }

        /// <summary>
        /// Creates a Continuous <see cref="ActionSpec"/> with the number of actions available.
        /// </summary>
        /// <param name="numActions">The number of actions available.</param>
        /// <returns>An Continuous ActionSpec initialized with the number of actions available.</returns>
        public static ActionSpec MakeContinuous(int numActions)
        {
            var actuatorSpace = new ActionSpec(numActions, null);
            return actuatorSpace;
        }

        /// <summary>
        /// Creates a Discrete <see cref="ActionSpec"/> with the array of branch sizes that
        /// represents the action space.
        /// </summary>
        /// <param name="branchSizes">The array of branch sizes for the discrete action space.  Each index
        /// contains the number of actions available for that branch.</param>
        /// <returns>An Discrete ActionSpec initialized with the array of branch sizes.</returns>
        public static ActionSpec MakeDiscrete(params int[] branchSizes)
        {
            var numActions = branchSizes.Length;
            var actuatorSpace = new ActionSpec(0, branchSizes);
            return actuatorSpace;
        }

        internal ActionSpec(int numContinuousActions, int[] branchSizes = null)
        {
            m_NumContinuousActions = numContinuousActions;
            BranchSizes = branchSizes;
        }

        /// <summary>
        /// Check that the ActionSpec uses either all continuous or all discrete actions.
        /// This is only used when connecting to old versions of the trainer that don't support this.
        /// </summary>
        /// <exception cref="UnityAgentsException"></exception>
        internal void CheckAllContinuousOrDiscrete()
        {
            if (NumContinuousActions > 0 && NumDiscreteActions > 0)
            {
                throw new UnityAgentsException(
                    "Action spaces with both continuous and discrete actions are not supported by the trainer. " +
                    "ActionSpecs must be all continuous or all discrete."
                );
            }
        }
    }
}
