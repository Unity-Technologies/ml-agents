namespace MLAgents.CoreInternalBrain
{
    public class NodeNames
    {
        public string BatchSizePlaceholder = "batch_size";
        public string SequenceLengthPlaceholder = "sequence_length";
        public string VectorObservationPlacholder = "vector_observation";
        public string RecurrentInPlaceholder = "recurrent_in";
        public string VisualObservationPlaceholderPrefix = "visual_observation_";
        public string PreviousActionPlaceholder = "prev_action";
        public string ActionMaskPlaceholder = "action_masks";
        public string RandomNormalEpsilonPlaceholder = "random_normal_epsilon";

        public string ValueEstimateOutput = "value_estimate";
        public string RecurrentOutOutput = "recurrent_out";
        public string MemorySize = "memory_size";
        public string kApiVersion = "api_version";
        public string ActionOutput = "action";
    }
}