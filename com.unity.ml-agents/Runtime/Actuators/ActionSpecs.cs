using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Defines the structure of an Action Space to be used by the Actuator system.
    /// </summary>
    public readonly struct ActionSpecs
    {

        /// <summary>
        /// An array of branch sizes for our action space.
        ///
        /// For an IActuator that uses a Discrete <see cref="SpaceType"/>, the number of
        /// branches is the Length of the Array and each index contains the branch size.
        /// The cumulative sum of the total number of discrete actions can be retrieved
        /// by the <see cref="SumOfDiscreteBranchSizes"/> property.
        ///
        /// For an IActuator with a Continuous <see cref="SpaceType"/>, the number of continuous
        /// actions is at the 0th index of array.
        /// </summary>
        public readonly int[] BranchSizes;

        /// <summary>
        /// The number of actions for a Continuous <see cref="SpaceType"/>.
        /// </summary>
        public readonly int NumContinuousActions;

        /// <summary>
        /// The number of branches for a Discrete <see cref="SpaceType"/>.
        /// </summary>
        public readonly int NumDiscreteActions;

        /// <summary>
        /// Get the total number of Discrete Actions that can be taken by calculating the Sum
        /// of all of the Discrete Action branch sizes.
        /// </summary>
        public int SumOfDiscreteBranchSizes { get; }

        /// <summary>
        /// Creates a Continuous <see cref="ActionSpecs"/> with the number of actions available.
        /// </summary>
        /// <param name="numActions">The number of actions available.</param>
        /// <returns>An Continuous ActionSpecs initialized with the number of actions available.</returns>
        public static ActionSpecs MakeContinuous(int numActions)
        {
            var actuatorSpace = new ActionSpecs(numActions, 0);
            return actuatorSpace;
        }

        /// <summary>
        /// Creates a Discrete <see cref="ActionSpecs"/> with the array of branch sizes that
        /// represents the action space.
        /// </summary>
        /// <param name="branchSizes">The array of branch sizes for the discrete action space.  Each index
        /// contains the number of actions available for that branch.</param>
        /// <returns>An Discrete ActionSpecs initialized with the array of branch sizes.</returns>
        public static ActionSpecs MakeDiscrete(int[] branchSizes)
        {
            var numActions = branchSizes.Length;
            var actuatorSpace = new ActionSpecs(0, numActions, branchSizes);
            return actuatorSpace;
        }

        ActionSpecs(int numContinuousActions, int numDiscreteActions, int[] branchSizes = null)
        {
            NumContinuousActions = numContinuousActions;
            NumDiscreteActions = numDiscreteActions;
            BranchSizes = branchSizes ?? new[] { numContinuousActions };
            SumOfDiscreteBranchSizes = branchSizes?.Sum() ?? 0;
        }
    }
}
