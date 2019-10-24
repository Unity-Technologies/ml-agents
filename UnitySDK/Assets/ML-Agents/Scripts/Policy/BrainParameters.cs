using System;
using UnityEngine;

namespace MLAgents
{
    public enum SpaceType
    {
        Discrete,
        Continuous
    }

    /// <summary>
    /// Holds information about the Brain. It defines what are the inputs and outputs of the
    /// decision process.
    /// </summary>
    [Serializable]
    public class BrainParameters
    {
        /// <summary>
        /// If continuous : The length of the float vector that represents
        /// the state
        /// If discrete : The number of possible values the state can take
        /// </summary>
        public int vectorObservationSize = 1;

        [Range(1, 50)] public int numStackedVectorObservations = 1;

        /// <summary>
        /// If continuous : The length of the float vector that represents
        /// the action
        /// If discrete : The number of possible values the action can take*/
        /// </summary>
        public int[] vectorActionSize = new[] {1};

        /// <summary></summary>The list of strings describing what the actions correpond to */
        public string[] vectorActionDescriptions;

        /// <summary>Defines if the action is discrete or continuous</summary>
        public SpaceType vectorActionSpaceType = SpaceType.Discrete;

        /// <summary>
        /// Deep clones the BrainParameter object
        /// </summary>
        /// <returns> A new BrainParameter object with the same values as the original.</returns>
        public BrainParameters Clone()
        {
            return new BrainParameters
            {
                vectorObservationSize = vectorObservationSize,
                numStackedVectorObservations = numStackedVectorObservations,
                vectorActionSize = (int[])vectorActionSize.Clone(),
                vectorActionDescriptions = (string[])vectorActionDescriptions.Clone(),
                vectorActionSpaceType = vectorActionSpaceType
            };
        }
    }
}
