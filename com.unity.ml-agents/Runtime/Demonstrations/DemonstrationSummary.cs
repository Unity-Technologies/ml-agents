using System;
using System.Collections.Generic;
using UnityEngine;
using MLAgents.Policies;

namespace MLAgents.Demonstrations
{
    /// <summary>
    /// Demonstration Object. Contains meta-data regarding demonstration.
    /// Used for imitation learning, or other forms of learning from data.
    /// </summary>
    [Serializable]
    internal class DemonstrationSummary : ScriptableObject
    {
        public DemonstrationMetaData metaData;
        public BrainParameters brainParameters;
        public List<int[]> observationShapes;

        public void Initialize(BrainParameters brainParams,
            DemonstrationMetaData demonstrationMetaData, List<int[]> obsShapes)
        {
            brainParameters = brainParams;
            metaData = demonstrationMetaData;
            observationShapes = obsShapes;
        }
    }
}
