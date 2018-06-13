using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

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
        public struct KeyContinuousPlayerAction
        {
            public KeyCode key;
            public int index;
            public float value;
        }
        
        [System.Serializable]
        public struct AxisContinuousPlayerAction
        {
            public string axis;
            public int index;
            public float scale;
        }

        [SerializeField]
        [FormerlySerializedAs("continuousPlayerActions")]
        [Tooltip("The list of keys and the value they correspond to for continuous control.")]
        /// Contains the mapping from input to continuous actions
        public KeyContinuousPlayerAction[] keyContinuousPlayerActions;
        
        [SerializeField]
        [Tooltip("The list of axis actions.")]
        /// Contains the mapping from input to continuous actions
        public AxisContinuousPlayerAction[] axisContinuousPlayerActions;
        
        [SerializeField]
        [Tooltip("The list of keys and the value they correspond to for discrete control.")]
        /// Contains the mapping from input to discrete actions
        public DiscretePlayerAction[] discretePlayerActions;
        [SerializeField]
        public int defaultAction = 0;

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

            if (brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                foreach (Agent agent in agentInfo.Keys)
                {
                    var action = new float[brainParameters.vectorActionSize];
                    foreach (KeyContinuousPlayerAction cha in keyContinuousPlayerActions)
                    {
                        if (Input.GetKey(cha.key))
                        {
                            action[cha.index] = cha.value;
                        }
                    }


                    foreach (AxisContinuousPlayerAction axisAction in axisContinuousPlayerActions)
                    {
                        var axisValue = Input.GetAxis(axisAction.axis);
                        axisValue *= axisAction.scale;
                        if (Mathf.Abs(axisValue) > 0.0001)
                        {
                            action[axisAction.index] = axisValue;
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
        }
    }

}
