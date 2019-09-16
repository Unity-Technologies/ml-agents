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
        [SerializeField]
        public List<Brain> broadcastingBrains = new List<Brain>();
        [FormerlySerializedAs("_brainsToControl")]
        [SerializeField]
        private List<Brain> m_BrainsToControl = new List<Brain>();

        /// <summary>
        /// The number of Brains inside the BroadcastingHub.
        /// </summary>
        public int Count
        {
            get { return broadcastingBrains.Count; }
        }

        /// <summary>
        /// Checks that a given Brain is set to be remote controlled.
        /// </summary>
        /// <param name="brain"> The Brain that is beeing checked</param>
        /// <returns>true if the Brain is set to Controlled and false otherwise. Will return
        /// false if the Brain is not present in the Hub.</returns>
        public bool IsControlled(Brain brain)
        {
            return m_BrainsToControl.Contains(brain);
        }

        /// <summary>
        /// Sets a brain to controlled.
        /// </summary>
        /// <param name="brain"> The Brain that is being set to controlled</param>
        /// <param name="controlled"> if true, the Brain will be set to remote controlled. Otherwise
        /// the brain will be set to broadcast only.</param>
        public void SetControlled(Brain brain, bool controlled)
        {
            if (broadcastingBrains.Contains(brain))
            {
                if (controlled && !m_BrainsToControl.Contains(brain))
                {
                    m_BrainsToControl.Add(brain);
                }

                if (!controlled && m_BrainsToControl.Contains(brain))
                {
                    m_BrainsToControl.Remove(brain);
                }
            }
        }

        /// <summary>
        /// Removes all the Brains of the BroadcastHub
        /// </summary>
        public void Clear()
        {
            broadcastingBrains.Clear();
            m_BrainsToControl.Clear();
        }
    }
}
