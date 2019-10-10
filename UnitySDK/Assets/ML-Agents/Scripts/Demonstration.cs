using System;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Object. Contains meta-data regarding demonstration.
    /// Used for imitation learning, or other forms of learning from data.
    /// </summary>
    [Serializable]
    public class Demonstration : ScriptableObject
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
    public class DemonstrationMetaData
    {
        public int numberExperiences;
        public int numberEpisodes;
        public float meanReward;
        public string demonstrationName;
        public const int ApiVersion = 1;
    }
}
