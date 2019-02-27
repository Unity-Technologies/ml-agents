namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Contains the names of the input and output Tensor for the Inference Brain.
    /// </summary>
    public static class TensorNames
    {
        public const string BatchSizePlaceholder = "batch_size";
        public const string SequenceLengthPlaceholder = "sequence_length";
        public const string VectorObservationPlacholder = "vector_observation";
        public const string RecurrentInPlaceholder = "recurrent_in";
        public const string RecurrentInPlaceholder_H = "recurrent_in_h";
        public const string RecurrentInPlaceholder_C = "recurrent_in_c";
        public const string VisualObservationPlaceholderPrefix = "visual_observation_";
        public const string PreviousActionPlaceholder = "prev_action";
        public const string ActionMaskPlaceholder = "action_masks";
        public const string RandomNormalEpsilonPlaceholder = "epsilon";

        public const string ValueEstimateOutput = "value_estimate";
        public const string RecurrentOutput = "recurrent_out";
        public const string RecurrentOutput_H = "recurrent_out_h";
        public const string RecurrentOutput_C = "recurrent_out_c";
        public const string MemorySize = "memory_size";
        public const string VersionNumber = "version_number";
        public const string IsContinuousControl = "is_continuous_control";
        public const string ActionOutputShape = "action_output_shape";
        public const string ActionOutput = "action";
    }
}
