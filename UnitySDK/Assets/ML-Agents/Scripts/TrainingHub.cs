using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{

    [System.Serializable]
    public class TrainingHub
    {
        [SerializeField]
        public List<Brain> exposedBrains = new List<Brain>();
        [SerializeField]
        private List<Brain> _brainsToTrain = new List<Brain>();

        public int Count
        {
            get { return exposedBrains.Count; }
        }        

        public bool IsTraining(Brain b)
        {
            return _brainsToTrain.Contains(b);
        }
        
        public void SetTraining(Brain b, bool train)
        {
            if (exposedBrains.Contains(b))
            {
                if (train && !_brainsToTrain.Contains(b))
                {
                    _brainsToTrain.Add(b);
                }

                if (!train && _brainsToTrain.Contains(b))
                {
                    _brainsToTrain.Remove(b);
                }
            }
        }
        
        public void Clear()
        {
            exposedBrains.Clear();
            _brainsToTrain.Clear();
        }
        

    }
}