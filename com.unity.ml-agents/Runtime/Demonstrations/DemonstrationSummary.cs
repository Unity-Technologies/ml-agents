using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Demonstrations
{
    /// <summary>
    /// Summary of a loaded Demonstration file. Only used for display in the Inspector.
    /// </summary>
    [Serializable]
    internal class DemonstrationSummary : ScriptableObject
    {
        public DemonstrationMetaData metaData;
        public BrainParameters brainParameters;
        public List<ObservationSummary> observationSummaries;

        public void Initialize(BrainParameters brainParams,
            DemonstrationMetaData demonstrationMetaData, List<ObservationSummary> obsSummaries)
        {
            brainParameters = brainParams;
            metaData = demonstrationMetaData;
            observationSummaries = obsSummaries;
        }
    }


    /// <summary>
    /// Summary of a loaded Observation. Currently only contains the shape of the Observation.
    /// </summary>
    /// <remarks>This is necessary because serialization doesn't support nested containers or arrays.</remarks>
    [Serializable]
    internal struct ObservationSummary
    {
        public int[] shape;
    }
}
