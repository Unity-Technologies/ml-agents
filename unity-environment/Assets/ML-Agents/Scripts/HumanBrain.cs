using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{


    public class HumanBrain : Brain
    {

        [System.Serializable]
        public struct DiscretePlayerAction
        {
            public KeyCode key;
            public int value;
        }

        [System.Serializable]
        public struct ContinuousPlayerAction
        {
            public KeyCode key;
            public int index;
            public float value;
        }

        [SerializeField]
        [Tooltip("The list of keys and the value they correspond to for continuous control.")]
        /// Contains the mapping from input to continuous actions
        public ContinuousPlayerAction[] continuousPlayerActions = new ContinuousPlayerAction[0];

        [SerializeField]
        [Tooltip("The list of keys and the value they correspond to for discrete control.")]
        /// Contains the mapping from input to discrete actions
        public DiscretePlayerAction[] discretePlayerActions = new DiscretePlayerAction[0];

        [SerializeField] public int defaultAction = 0;
        
        /**< Reference to the Decision component used to decide the actions */
//        public Decision decision = new RandomDecision();

//        public override void InitializeBrain(Academy aca, MLAgents.Batcher batcher, bool external)
           //        {
           //            aca.BrainDecideAction += DecideAction;
           //            if (batcher == null)
           //            {
           //                this.brainBatcher = null;
           //            }
           //            else
           //            {
           //                this.brainBatcher = batcher;
           //                this.brainBatcher.SubscribeBrain(this.name);
           //            }
           //        }

        /// Uses the continuous inputs or dicrete inputs of the player to 
        /// decide action
        protected override void DecideAction()
        {
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(this.name, agentInfo);
            }

            if (isExternal)
            {
                agentInfo.Clear();
                return;
            }

            if (this.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                foreach (Agent agent in agentInfo.Keys)
                {
                    var action = new float[this.brainParameters.vectorActionSize];
                    foreach (ContinuousPlayerAction cha in continuousPlayerActions)
                    {
                        if (Input.GetKey(cha.key))
                        {
                            action[cha.index] = cha.value;
                        }
                    }

                    agent.UpdateVectorAction(action);
                }

            }
            else
            {
                foreach (Agent agent in agentInfo.Keys)
                {
                    var action = new float[1] {defaultAction};
                    foreach (DiscretePlayerAction dha in discretePlayerActions)
                    {
                        if (Input.GetKey(dha.key))
                        {
                            action[0] = (float) dha.value;
                            break;
                        }
                    }


                    agent.UpdateVectorAction(action);

                }
            }
            agentInfo.Clear();

        }
    }

}
