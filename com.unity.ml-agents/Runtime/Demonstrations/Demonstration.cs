using System;
using UnityEngine;
using MLAgents.Policies;

namespace MLAgents.Demonstrations
{
    /// <summary>
    /// Demonstration Object. Contains meta-data regarding demonstration.
    /// Used for imitation learning, or other forms of learning from data.
    /// </summary>
    [Serializable]
    internal class Demonstration : ScriptableObject
    {
        public DemonstrationMetaData metaData;
        public BrainParameters brainParameters;

        public void Initialize(BrainParameters brainParams,
            DemonstrationMetaData demonstrationMetaData)
        {
            brainParameters = brainParams;
            metaData = demonstrationMetaData;
        }
    }

    /// <summary>
    /// Demonstration meta-data.
    /// Kept in a struct for easy serialization and deserialization.
    /// </summary>
    [Serializable]
    internal class DemonstrationMetaData
    {
        public int numberExperiences;
        public int numberEpisodes;
        public float meanReward;
        public string demonstrationName;
        public const int ApiVersion = 1;
    }
}
