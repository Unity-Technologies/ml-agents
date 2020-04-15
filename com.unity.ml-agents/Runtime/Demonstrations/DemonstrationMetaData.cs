using System;
using UnityEngine;
using MLAgents.Policies;
using UnityEngine.Serialization;

namespace MLAgents.Demonstrations
{
    /// <summary>
    /// Demonstration meta-data.
    /// Kept in a struct for easy serialization and deserialization.
    /// </summary>
    [Serializable]
    internal class DemonstrationMetaData
    {
        [FormerlySerializedAs("numberExperiences")]
        public int numberSteps;
        public int numberEpisodes;
        public float meanReward;
        public string demonstrationName;
        public const int ApiVersion = 1;
    }
}
