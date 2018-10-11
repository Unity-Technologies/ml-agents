using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Brain receive data from Agents through calls to SendState. The brain then updates the
    /// actions of the agents at each FixedUpdate.
    /// The Brain encapsulates the decision making process. Every Agent must be assigned a Brain,
    /// but you can use the same Brain with more than one Agent. You can also create several
    /// Brains, attach each of the Brain to one or more than one Agent.
    /// Brain assets has several important properties that you can set using the Inspector window.
    /// These properties must be appropriate for the Agents using the Brain. For example, the
    /// Vector Observation Space Size property must match the length of the feature
    /// vector created by an Agent exactly.
    /// </summary>
    public abstract class Brain : ScriptableObject
    {
        [SerializeField] public BrainParameters brainParameters;

        protected Dictionary<Agent, AgentInfo> agentInfos =
            new Dictionary<Agent, AgentInfo>(1024);

        protected Batcher brainBatcher;

        [System.NonSerialized]
        private bool _isInitialized;

        /// <summary>
        /// Sets the Batcher of the Brain. The brain will call the batcher at every step and give
        /// it the agent's data using SendBrainInfo at each DecideAction call.
        /// </summary>
        /// <param name="batcher"> The Batcher the brain will use for the current session</param>
        public void SetBatcher(Batcher batcher)
        {
            if (batcher == null)
            {
                brainBatcher = null;
            }
            else
            {
                brainBatcher = batcher;
                brainBatcher.SubscribeBrain(name);
            }
            LazyInitialize();
        }
        
        /// <summary>
        /// Adds the data of an agent to the current batch so it will be processed in DecideAction.
        /// </summary>
        /// <param name="agent"></param>
        /// <param name="info"></param>
        public void SendState(Agent agent, AgentInfo info)
        {
            LazyInitialize();
            agentInfos.Add(agent, info);

        }

        /// <summary>
        /// If the Brain is not initialized, it subscribes to the Academy's DecideAction Event and
        /// calls the Initialize method to be implemented by child classes.
        /// </summary>
        private void LazyInitialize()
        {
            if (!_isInitialized)
            {
                FindObjectOfType<Academy>().BrainDecideAction += BrainDecideAction;
                Initialize();
                _isInitialized = true;
            }
        }
        
        /// <summary>
        /// Calls the DecideAction method that the concrete brain implements.
        /// </summary>
        private void BrainDecideAction()
        {
            brainBatcher?.SendBrainInfo(name, agentInfos);
            DecideAction();
        }

        /// <summary>
        /// Is called only once at the begening of the training or inference session.
        /// </summary>
        protected abstract void Initialize();

        /// <summary>
        /// Is called once per Environment Step after the Brain has been initialized.
        /// </summary>
        protected abstract void DecideAction();
    }
}
