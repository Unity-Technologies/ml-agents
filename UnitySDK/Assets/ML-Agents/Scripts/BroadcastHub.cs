using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{

    [System.Serializable]
    public class BroadcastHub
    {
        [SerializeField]
        public List<Brain> broadcastingBrains = new List<Brain>();
        [SerializeField]
        private List<Brain> _brainsToControl = new List<Brain>();

        public int Count
        {
            get { return broadcastingBrains.Count; }
        }        

        public bool IsControlled(Brain b)
        {
            return _brainsToControl.Contains(b);
        }
        
        public void SetTraining(Brain b, bool train)
        {
            if (broadcastingBrains.Contains(b))
            {
                if (train && !_brainsToControl.Contains(b))
                {
                    _brainsToControl.Add(b);
                }

                if (!train && _brainsToControl.Contains(b))
                {
                    _brainsToControl.Remove(b);
                }
            }
        }
        
        public void Clear()
        {
            broadcastingBrains.Clear();
            _brainsToControl.Clear();
        }
        

    }
}