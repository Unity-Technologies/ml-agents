using System;
using MLAgents.CommunicatorObjects;
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

        /// <summary>
        /// Constructor for initializing metadata to default values.
        /// </summary>
        public DemonstrationMetaData()
        {
        }

        /// <summary>
        /// Initialize metadata values based on proto object.
        /// </summary>
        public DemonstrationMetaData(DemonstrationMetaProto demoProto)
        {
            numberEpisodes = demoProto.NumberEpisodes;
            numberExperiences = demoProto.NumberSteps;
            meanReward = demoProto.MeanReward;
            demonstrationName = demoProto.DemonstrationName;
            if (demoProto.ApiVersion != ApiVersion)
            {
                throw new Exception("API versions of demonstration are incompatible.");
            }
        }

        /// <summary>
        /// Convert metadata object to proto object.
        /// </summary>
        public DemonstrationMetaProto ToProto()
        {
            var demoProto = new DemonstrationMetaProto
            {
                ApiVersion = ApiVersion,
                MeanReward = meanReward,
                NumberSteps = numberExperiences,
                NumberEpisodes = numberEpisodes,
                DemonstrationName = demonstrationName
            };
            return demoProto;
        }
    }
}
