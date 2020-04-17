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
    /// Holds information about the brain. It defines what are the inputs and outputs of the
    /// decision process.
    /// </summary>
    /// <remarks>
    /// Set brain parameters for an <see cref="Agent"/> instance using the
    /// <seealso cref="BehaviorParameters"/> component attached to the agent's [GameObject].
    ///
    /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
    /// </remarks>
    [Serializable]
    public class BrainParameters
    {
        /// <summary>
        /// The size of the observation space.
        /// </summary>
        /// <remarks>An agent creates the observation vector in its
        /// <see cref="Agent.CollectObservations(Sensors.VectorSensor)"/>
        /// implementation.</remarks>
        /// <value>
        /// The length of the vector containing observation values.
        /// </value>
        public int vectorObservationSize = 1;

        /// <summary>
        /// Stacking refers to concatenating the observations across multiple frames. This field
        /// indicates the number of frames to concatenate across.
        /// </summary>
        [Range(1, 50)] public int numStackedVectorObservations = 1;

        /// <summary>
        /// The size of the action space.
        /// </summary>
        /// <remarks>The size specified is interpreted differently depending on whether
        /// the agent uses the continuous or the discrete action space.</remarks>
        /// <value>
        /// For the continuous action space: the length of the float vector that represents
        /// the action.
        /// For the discrete action space: the number of branches in the action space.
        /// </value>
        public int[] vectorActionSize = new[] {1};

        /// <summary>
        /// The list of strings describing what the actions correspond to.
        /// </summary>
        public string[] vectorActionDescriptions;

        /// <summary>
        /// Defines if the action is discrete or continuous.
        /// </summary>
        public SpaceType vectorActionSpaceType = SpaceType.Discrete;

        /// <summary>
        /// The effective action space size.
        /// </summary>
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
