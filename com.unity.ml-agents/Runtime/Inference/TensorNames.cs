namespace MLAgents.Inference
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
        public const string recurrentInPlaceholderH = "recurrent_in_h";
        public const string recurrentInPlaceholderC = "recurrent_in_c";
        public const string VisualObservationPlaceholderPrefix = "visual_observation_";
        public const string PreviousActionPlaceholder = "prev_action";
        public const string ActionMaskPlaceholder = "action_masks";
        public const string RandomNormalEpsilonPlaceholder = "epsilon";

        public const string ValueEstimateOutput = "value_estimate";
        public const string RecurrentOutput = "recurrent_out";
        public const string recurrentOutputH = "recurrent_out_h";
        public const string recurrentOutputC = "recurrent_out_c";
        public const string MemorySize = "memory_size";
        public const string VersionNumber = "version_number";
        public const string IsContinuousControl = "is_continuous_control";
        public const string ActionOutputShape = "action_output_shape";
        public const string ActionOutput = "action";
    }
}
