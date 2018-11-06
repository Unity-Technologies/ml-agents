using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// Implemetation of the Player Brain. Inherits from the base class Brain. Allows the user to
    /// manually select decisions for linked agents by creating a mapping from keys presses to
    /// actions.
    /// You can use Player Brains to control a "teacher" Agent that trains other Agents during
    /// imitation learning. You can also use Player Brains to test your Agents and environment
    /// before training agents with reinforcement learning.
    /// </summary>
    [CreateAssetMenu(fileName = "NewPlayerBrain", menuName = "ML-Agents/Player Brain")]
    public class PlayerBrain : Brain
    {
        [System.Serializable]
        public struct DiscretePlayerAction
        {
            public KeyCode key;
            public int branchIndex;
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

        protected override void Initialize(){ }
        
        /// Uses the continuous inputs or dicrete inputs of the player to 
        /// decide action
        protected override void DecideAction()
        {
            if (brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                foreach (Agent agent in agentInfos.Keys)
                {
                    var action = new float[brainParameters.vectorActionSize[0]];
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
                foreach (Agent agent in agentInfos.Keys)
                {
                    var action = new float[brainParameters.vectorActionSize.Length];
                    foreach (DiscretePlayerAction dha in discretePlayerActions)
                    {
                        if (Input.GetKey(dha.key))
                        {
                            action[dha.branchIndex] = (float) dha.value;
                        }
                    }
                    agent.UpdateVectorAction(action);
                }
            }
            agentInfos.Clear();
        }
    }
}
