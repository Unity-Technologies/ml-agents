using System;
using System.Collections.Generic;

namespace Unity.MLAgents.Analytics
{
    internal struct InferenceEvent
    {
        public string BarracudaModelSource;
        public string BarracudaModelVersion;
        public string BarracudaModelProducer;
        public string BarracudaPackageVersion;
        public int InferenceDevice;
        public EventObservationSpec ObservationSpec;
        public EventActionSpec ActionSpec;
        public int MemorySize;
        // public Int64 ModelHash; TODO ?
    }

    /// <summary>
    /// Simplified version of ActionSpec struct for use in analytics
    /// </summary>
    internal struct EventActionSpec
    {
        public int NumContinuousActions;
        public int NumDiscreteActions;
    }

    /// <summary>
    /// Simplified summary of Agent observations for use in analytics
    /// </summary>
    internal struct EventObservationSpec
    {
        public int NumVectorObservations;
        public List<List<int>> VisualObservationSizes;
    }
}
