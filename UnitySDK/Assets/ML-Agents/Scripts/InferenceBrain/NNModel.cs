using UnityEngine;

namespace MLAgents.InferenceBrain
{
    public class NNModel : ScriptableObject
    {
        [HideInInspector]
        public byte[] Value;
    }
}
