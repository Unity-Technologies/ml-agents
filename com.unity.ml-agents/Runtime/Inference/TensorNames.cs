using System.Collections.Generic;
using System.Linq;
using System;
namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Contains the names of the input and output tensors for the Inference Brain.
    /// </summary>
    internal static class TensorNames
    {
        public const string BatchSizePlaceholder = "batch_size";
        public const string SequenceLengthPlaceholder = "sequence_length";
        public const string VectorObservationPlaceholder = "vector_observation";
        public const string RecurrentInPlaceholder = "recurrent_in";
        public const string VisualObservationPlaceholderPrefix = "visual_observation_";
        public const string ObservationPlaceholderPrefix = "iobs_";
        public const string PreviousActionPlaceholder = "prev_action";
        public const string ActionMaskPlaceholder = "action_masks";
        public const string RandomNormalEpsilonPlaceholder = "epsilon";

        public const string ValueEstimateOutput = "value_estimate";
        public const string RecurrentOutput = "recurrent_out";
        public const string MemorySize = "memory_size";
        public const string VersionNumber = "version_number";
        public const string ContinuousActionOutputShape = "continuous_action_output_shape";
        public const string DiscreteActionOutputShape = "discrete_action_output_shape";
        public const string ContinuousActionOutput = "continuous_actions";
        public const string DiscreteActionOutput = "discrete_actions";

        // Deprecated TensorNames entries for backward compatibility
        public const string IsContinuousControlDeprecated = "is_continuous_control";
        public const string ActionOutputDeprecated = "action_";
        public const string ActionOutputShapeDeprecated = "action_output_shape";

        // Tensors for in-editor training
        public const string Observations = "obs_0";
        public const string ActionInput = "action";
        public const string RewardInput = "reward";
        public const string DoneInput = "done";
        public const string Gamma = "gamma";
        public const string NextObservations = "next_obs_0";
        public const string LearningRate = "lr";
        public const string TrainingStateIn = "training_state.1";


        public const string TrainingOutput = "output";
        public const string OuputLoss = "loss";
        public const string TrainingStateOut = "training_state";
        public const string InitialTrainingState = "initial_training_state";

        /// <summary>
        /// Returns the name of the visual observation with a given index
        /// </summary>
        public static string GetVisualObservationName(int index)
        {
            return VisualObservationPlaceholderPrefix + index;
        }

        /// <summary>
        /// Returns the name of the observation with a given index
        /// </summary>
        public static string GetObservationName(int index)
        {
            return ObservationPlaceholderPrefix + index;
        }
    }
}
