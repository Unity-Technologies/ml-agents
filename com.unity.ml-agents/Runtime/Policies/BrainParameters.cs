using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Policies
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
        /// The number of the observations that are added in
        /// <see cref="Agent.CollectObservations(Sensors.VectorSensor)"/>
        /// </summary>
        /// <value>
        /// The length of the vector containing observation values.
        /// </value>
        [FormerlySerializedAs("vectorObservationSize")]
        public int VectorObservationSize = 1;

        /// <summary>
        /// Stacking refers to concatenating the observations across multiple frames. This field
        /// indicates the number of frames to concatenate across.
        /// </summary>
        [FormerlySerializedAs("numStackedVectorObservations")]
        [Range(1, 50)] public int NumStackedVectorObservations = 1;

        public ActionSpec VectorActionSpec = new ActionSpec(0, 0, new int[] { });

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
        [FormerlySerializedAs("vectorActionSize")]
        [FormerlySerializedAs("VectorActionSize")]
        public int[] VectorActionSizeDeprecated = new[] { 1 };

        /// <summary>
        /// The list of strings describing what the actions correspond to.
        /// </summary>
        [FormerlySerializedAs("vectorActionDescriptions")]
        public string[] VectorActionDescriptions;

        /// <summary>
        /// Defines if the action is discrete or continuous.
        /// </summary>
        [FormerlySerializedAs("vectorActionSpaceType")]
        [FormerlySerializedAs("VectorActionSpaceType")]
        public SpaceType VectorActionSpaceTypeDeprecated = SpaceType.Discrete;

        [SerializeField]
        [HideInInspector]
        internal bool hasUpgradedBrainParametersWithActionSpec;

        /// <summary>
        /// The number of actions specified by this Brain.
        /// </summary>
        public int NumActionsDeprecated
        {
            get
            {
                switch (VectorActionSpaceTypeDeprecated)
                {
                    case SpaceType.Discrete:
                        return VectorActionSizeDeprecated.Length;
                    case SpaceType.Continuous:
                        return VectorActionSizeDeprecated[0];
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
                VectorObservationSize = VectorObservationSize,
                NumStackedVectorObservations = NumStackedVectorObservations,
                VectorActionSizeDeprecated = (int[])VectorActionSizeDeprecated.Clone(),
                VectorActionDescriptions = (string[])VectorActionDescriptions.Clone(),
                VectorActionSpaceTypeDeprecated = VectorActionSpaceTypeDeprecated,
                VectorActionSpec = new ActionSpec(VectorActionSpec.NumContinuousActions, VectorActionSpec.NumDiscreteActions, VectorActionSpec.BranchSizes)
            };
        }
    }
}
