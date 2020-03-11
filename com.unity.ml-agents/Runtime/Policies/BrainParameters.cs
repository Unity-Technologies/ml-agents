using System;
using UnityEngine;

namespace MLAgents.Policies
{
    /// <summary>
    /// Whether the action space is discrete or continuous.
    /// </summary>
    public enum SpaceType
    {
        /// <summary>
        /// Discrete action space: a fixed number of options are available.
        /// </summary>
        Discrete,

        /// <summary>
        /// Continuous action space: each action can take on a float value.
        /// </summary>
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
        /// If continuous : The length of the float vector that represents the state.
        /// If discrete : The number of possible values the state can take.
        /// </summary>
        public int vectorObservationSize = 1;

        /// <summary>
        /// Stacking refers to concatenating the observations across multiple frames. This field
        /// indicates the number of frames to concatenate across.
        /// </summary>
        [Range(1, 50)] public int numStackedVectorObservations = 1;

        /// <summary>
        /// If continuous : The length of the float vector that represents the action.
        /// If discrete : The number of possible values the action can take.
        /// </summary>
        public int[] vectorActionSize = new[] {1};

        /// <summary>
        /// The list of strings describing what the actions correspond to.
        /// </summary>
        public string[] vectorActionDescriptions;

        /// <summary>
        /// Defines if the action is discrete or continuous.
        /// </summary>
        public SpaceType vectorActionSpaceType = SpaceType.Discrete;

        public int numActions
        {
            get
            {
                switch (vectorActionSpaceType)
                {
                    case SpaceType.Discrete:
                        return vectorActionSize.Length;
                    case SpaceType.Continuous:
                        return vectorActionSize[0];
                    default:
                        return 0;
                }
            }
        }

        /// <summary>
        /// Deep clones the BrainParameter object.
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
