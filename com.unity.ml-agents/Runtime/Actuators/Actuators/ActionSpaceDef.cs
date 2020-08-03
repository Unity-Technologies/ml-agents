using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    public readonly struct ActionSpaceDef
    {

        public readonly SpaceType SpaceType;

        /// <summary>
        /// An array of branch sizes for our action space.
        ///
        /// For an IActuator that uses a Discrete <see cref="SpaceType"/>, the number of
        /// branches is the Length of the Array and each index contains the branch size.
        /// The cumulative sum of the total number of discrete actions can be retrieved
        /// by the <see cref="sumOfDiscreteBrancheSizes"/> property.
        ///
        /// For an IActuator with a Continuous <see cref="SpaceType"/>, the number of continuous
        /// actions is at the 0th index of array.
        /// </summary>
        public readonly int[] BranchSizes;

        /// <summary>
        /// The number of branches for a Discrete <see cref="SpaceType"/>, or the number of actions
        /// for a Continuous <see cref="SpaceType"/>.
        /// </summary>
        public readonly int NumActions;

        /// <summary>
        /// Get the total number of Discrete Actions that can be taken by calculating the Sum
        /// of all of the Discrete Action branch sizes.
        /// </summary>
        public int sumOfDiscreteBrancheSizes { get; }

        /// <summary>
        /// Creates a Continuous <see cref="ActionSpaceDef"/> with the number of actions available.
        /// </summary>
        /// <param name="numActions">The number of actions available.</param>
        /// <returns>An Continuous ActuatorSpace initialized with the number of actions available.</returns>
        public static ActionSpaceDef MakeContinuous(int numActions)
        {
            var actuatorSpace = new ActionSpaceDef(SpaceType.Continuous, numActions);
            return actuatorSpace;
        }

        /// <summary>
        /// Creates a Discrete <see cref="ActionSpaceDef"/> with the array of branch sizes that
        /// represents the action space.
        /// </summary>
        /// <param name="branchSizes">The array of branch sizes for the discrete action space.  Each index
        /// contains the number of actions available for that branch.</param>
        /// <returns>An Discrete ActuatorSpace initialized with the array of branch sizes.</returns>
        public static ActionSpaceDef MakeDiscrete(int[] branchSizes)
        {
            var numActions = branchSizes.Length;
            var actuatorSpace = new ActionSpaceDef(SpaceType.Discrete, numActions, branchSizes);
            return actuatorSpace;
        }

        ActionSpaceDef(SpaceType spaceType, int numActions, int[] branchSizes = null)
        {
            SpaceType = spaceType;
            NumActions = numActions;
            BranchSizes = branchSizes;
            sumOfDiscreteBrancheSizes = BranchSizes?.Sum() ?? 0;
        }
    }
}
