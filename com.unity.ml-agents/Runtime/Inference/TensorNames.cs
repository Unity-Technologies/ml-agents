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
        public const string ObservationPlaceholderPrefix = "obs_";
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
        public const string ActionOutputDeprecated = "action";
        public const string ActionOutputShapeDeprecated = "action_output_shape";

        // Tensors for in-editor training
        public const string ActionInput = "action_in";
        public const string RewardInput = "reward";
        public const string NextObservationPlaceholderPrefix = "next_obs_";
        public const string TargetInput = "target";
        public const string LearningRate = "lr";
        public const string InputWeightsPrefix = "w_";
        public const string InputBiasPrefix = "b_";
        public const string OutputWeightsPrefix = "nw_";
        public const string OutputBiasPrefix = "nb_";

        /// <summary>
        /// Returns the name of the visual observation with a given index
        /// </summary>
        public static string GetVisualObservationName(int index)
        {
            return VisualObservationPlaceholderPrefix + index;
        }

        static HashSet<string> InferenceInput = new HashSet<string>
            {
                BatchSizePlaceholder,
                SequenceLengthPlaceholder,
                VectorObservationPlaceholder,
                RecurrentInPlaceholder,
                VisualObservationPlaceholderPrefix,
                ObservationPlaceholderPrefix,
                PreviousActionPlaceholder,
                ActionMaskPlaceholder,
                RandomNormalEpsilonPlaceholder
            };
        static HashSet<string> InferenceInputPrefix = new HashSet<string>
            {
                VisualObservationPlaceholderPrefix,
                ObservationPlaceholderPrefix,
            };
        static HashSet<string> TrainingInput = new HashSet<string>
            {
                ActionInput,
                RewardInput,
                TargetInput,
                LearningRate,
                BatchSizePlaceholder,
            };
        static HashSet<string> TrainingInputPrefix = new HashSet<string>
            {
                ObservationPlaceholderPrefix,
                NextObservationPlaceholderPrefix,
            };
         static HashSet<string> ModelParamPrefix = new HashSet<string>
            {
                InputWeightsPrefix,
                InputBiasPrefix,
            };

        /// <summary>
        /// Returns the name of the observation with a given index
        /// </summary>
        public static string GetObservationName(int index)
        {
            return ObservationPlaceholderPrefix + index;
        }
        public static string GetNextObservationName(int index)
        {
            return ObservationPlaceholderPrefix + index;
        }

        public static string GetInputWeightName(int index)
        {
            return InputWeightsPrefix + index;
        }

        public static string GetInputBiasName(int index)
        {
            return InputBiasPrefix + index;
        }

        public static bool IsInferenceInputNames(string name)
        {
            return InferenceInput.Contains(name) || InferenceInputPrefix.Any(s=>name.Contains(s));
        }
        public static bool IsTrainingInputNames(string name)
        {
            return TrainingInput.Contains(name) || TrainingInputPrefix.Any(s=>name.Contains(s));
        }
        public static bool IsModelParamNames(string name)
        {
            return ModelParamPrefix.Any(s=>name.Contains(s));
        }
    }
}
