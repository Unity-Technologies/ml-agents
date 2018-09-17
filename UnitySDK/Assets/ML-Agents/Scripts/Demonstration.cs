using UnityEngine;

namespace MLAgents
{
    public class Demonstration : ScriptableObject
    {
        [SerializeField] public string demonstrationName;
        
        [Header("Properties of demonstration")]
        [SerializeField] public BrainParameters brainParameters;
        [SerializeField] public DemonstrationMetaData metaData;
    }
}
