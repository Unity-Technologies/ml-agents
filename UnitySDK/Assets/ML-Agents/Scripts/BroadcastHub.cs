using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// BroadcastHub holds reference to brains and keeps track wether or not the brain be
    /// remotely controlled.
    /// </summary>
    [System.Serializable]
    public class BroadcastHub
    {
        [FormerlySerializedAs("_brainsToControl")]
        [SerializeField]
        public List<LearningBrain> brainsToControl = new List<LearningBrain>();

        /// <summary>
        /// The number of Brains inside the BroadcastingHub.
        /// </summary>
        public int Count
        {
            get { return brainsToControl.Count; }
        }

        /// <summary>
        /// Sets a brain to controlled.
        /// </summary>
        /// <param name="brain"> The Brain that is being set to controlled</param>
        public void SetControlled(LearningBrain brain)
        {
            if (!brainsToControl.Contains(brain))
            {
                brainsToControl.Add(brain);
            }
        }

        /// <summary>
        /// Removes all the Brains of the BroadcastHub
        /// </summary>
        public void Clear()
        {
            brainsToControl.Clear();
        }
    }
}
