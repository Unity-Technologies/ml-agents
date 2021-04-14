using System;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// This is deprecated. Agents can now use both continuous and discrete actions together.
    /// </summary>
    [Obsolete("Continuous and discrete actions on the same Agent are now supported; see ActionSpec.")]
    internal enum SpaceType
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
    public class BrainParameters : ISerializationCallbackReceiver
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

        [SerializeField]
        internal ActionSpec m_ActionSpec = new ActionSpec(0, null);

        /// <summary>
        /// The specification of the Actions for the BrainParameters.
        /// </summary>
        public ActionSpec ActionSpec
        {
            get { return m_ActionSpec; }
            set
            {
                m_ActionSpec.NumContinuousActions = value.NumContinuousActions;
                m_ActionSpec.BranchSizes = value.BranchSizes;
                SyncDeprecatedActionFields();
            }
        }

        /// <summary>
        /// (Deprecated) The number of possible actions.
        /// </summary>
        /// <remarks>The size specified is interpreted differently depending on whether
        /// the agent uses the continuous or the discrete actions.</remarks>
        /// <value>
        /// For the continuous actions: the length of the float vector that represents
        /// the action.
        /// For the discrete actions: the number of branches.
        /// </value>
        [Obsolete("VectorActionSize has been deprecated, please use ActionSpec instead.")]
        [SerializeField]
        [FormerlySerializedAs("vectorActionSize")]
        internal int[] VectorActionSize = new[] { 1 };

        /// <summary>
        /// The list of strings describing what the actions correspond to.
        /// </summary>
        [FormerlySerializedAs("vectorActionDescriptions")]
        public string[] VectorActionDescriptions;

        /// <summary>
        /// (Deprecated) Defines if the action is discrete or continuous.
        /// </summary>
        [Obsolete("VectorActionSpaceType has been deprecated, please use ActionSpec instead.")]
        [SerializeField]
        [FormerlySerializedAs("vectorActionSpaceType")]
        internal SpaceType VectorActionSpaceType = SpaceType.Discrete;

        [SerializeField]
        [HideInInspector]
        internal bool hasUpgradedBrainParametersWithActionSpec;

        /// <summary>
        /// Deep clones the BrainParameter object.
        /// </summary>
        /// <returns> A new BrainParameter object with the same values as the original.</returns>
        public BrainParameters Clone()
        {
            // Disable deprecation warnings so we can read/write the old fields.
#pragma warning disable CS0618
            return new BrainParameters
            {
                VectorObservationSize = VectorObservationSize,
                NumStackedVectorObservations = NumStackedVectorObservations,
                VectorActionDescriptions = (string[])VectorActionDescriptions.Clone(),
                ActionSpec = new ActionSpec(ActionSpec.NumContinuousActions, ActionSpec.BranchSizes),
                VectorActionSize = (int[])VectorActionSize.Clone(),
                VectorActionSpaceType = VectorActionSpaceType,
            };
#pragma warning restore CS0618
        }

        /// <summary>
        /// Propagate ActionSpec fields from deprecated fields
        /// </summary>
        private void UpdateToActionSpec()
        {
            // Disable deprecation warnings so we can read the old fields.
#pragma warning disable CS0618
            if (!hasUpgradedBrainParametersWithActionSpec
                && m_ActionSpec.NumContinuousActions == 0
                && m_ActionSpec.NumDiscreteActions == 0)
            {
                if (VectorActionSpaceType == SpaceType.Continuous)
                {
                    m_ActionSpec.NumContinuousActions = VectorActionSize[0];
                }
                if (VectorActionSpaceType == SpaceType.Discrete)
                {
                    m_ActionSpec.BranchSizes = (int[])VectorActionSize.Clone();
                }
            }
            hasUpgradedBrainParametersWithActionSpec = true;
#pragma warning restore CS0618
        }

        /// <summary>
        /// Sync values in ActionSpec fields to deprecated fields
        /// </summary>
        private void SyncDeprecatedActionFields()
        {
            // Disable deprecation warnings so we can read the old fields.
#pragma warning disable CS0618

            if (m_ActionSpec.NumContinuousActions == 0)
            {
                VectorActionSize = (int[])ActionSpec.BranchSizes.Clone();
                VectorActionSpaceType = SpaceType.Discrete;
            }
            else if (m_ActionSpec.NumDiscreteActions == 0)
            {
                VectorActionSize = new[] { m_ActionSpec.NumContinuousActions };
                VectorActionSpaceType = SpaceType.Continuous;
            }
            else
            {
                VectorActionSize = null;
            }
#pragma warning restore CS0618
        }

        /// <summary>
        /// Called by Unity immediately before serializing this object.
        /// </summary>
        public void OnBeforeSerialize()
        {
            UpdateToActionSpec();
            SyncDeprecatedActionFields();
        }

        /// <summary>
        /// Called by Unity immediately after deserializing this object.
        /// </summary>
        public void OnAfterDeserialize()
        {
            UpdateToActionSpec();
            SyncDeprecatedActionFields();
        }
    }
}
