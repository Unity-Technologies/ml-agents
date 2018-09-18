using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Object. Contains meta-data regarding demonstration.
    /// </summary>
    public class Demonstration : ScriptableObject
    {
        [SerializeField] public string demonstrationName;
        
        [Header("Properties of demonstration")]
        [SerializeField] public BrainParameters brainParameters;
        [SerializeField] public DemonstrationMetaData metaData;
    }
}
