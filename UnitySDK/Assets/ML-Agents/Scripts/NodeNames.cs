namespace MLAgents.InferenceBrain
{
    public static class NodeNames
    {
        public const string BatchSizePlaceholder = "batch_size";
        public const string SequenceLengthPlaceholder = "sequence_length";
        public const string VectorObservationPlacholder = "vector_observation";
        public const string RecurrentInPlaceholder = "recurrent_in";
        public const string VisualObservationPlaceholderPrefix = "visual_observation_";
        public const string PreviousActionPlaceholder = "prev_action";
        public const string ActionMaskPlaceholder = "action_masks";
        public const string RandomNormalEpsilonPlaceholder = "epsilon";

        public const string ValueEstimateOutput = "value_estimate";
        public const string RecurrentOutOutput = "recurrent_out";
        public const string MemorySize = "memory_size";
        public const string VersionNumber = "version_number";
        public const string IsContinuousControl = "is_continuous_control";
        public const string ActionOutput = "action";
    }
}
