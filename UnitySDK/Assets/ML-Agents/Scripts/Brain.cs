using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Brain receive data from Agents through calls to SubscribeAgentForDecision. The brain then updates the
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

        /// <summary>
        /// List of agents subscribed for decisions.
        /// </summary>
        protected List<Agent> m_Agents = new List<Agent>(1024);

        protected ICommunicator m_Communicator;

        [NonSerialized]
        private bool m_IsInitialized;

        /// <summary>
        /// Sets the Batcher of the Brain. The brain will call the communicator at every step and give
        /// it the agent's data using PutObservations at each DecideAction call.
        /// </summary>
        /// <param name="communicator"> The Batcher the brain will use for the current session</param>
        public void SetCommunicator(ICommunicator communicator)
        {
            m_Communicator = communicator;
            LazyInitialize();
        }

        /// <summary>
        /// Registers an agent to current batch so it will be processed in DecideAction.
        /// </summary>
        /// <param name="agent"></param>
        public void SubscribeAgentForDecision(Agent agent)
        {
            LazyInitialize();
            m_Agents.Add(agent);
        }

        /// <summary>
        /// If the Brain is not initialized, it subscribes to the Academy's DecideAction Event and
        /// calls the Initialize method to be implemented by child classes.
        /// </summary>
        private void LazyInitialize()
        {
            if (!m_IsInitialized)
            {
                var academy = FindObjectOfType<Academy>();
                if (academy)
                {
                    academy.BrainDecideAction += BrainDecideAction;
                    academy.DestroyAction += Shutdown;
                    Initialize();
                    m_IsInitialized = true;
                }
            }
        }

        /// <summary>
        /// Called by the Academy when it shuts down. This ensures that the Brain cleans up properly
        /// after scene changes.
        /// </summary>
        private void Shutdown()
        {
            if (m_IsInitialized)
            {
                m_Agents.Clear();
                m_IsInitialized = false;
            }
        }

        /// <summary>
        /// Calls the DecideAction method that the concrete brain implements.
        /// </summary>
        private void BrainDecideAction()
        {
            m_Communicator?.PutObservations(name, m_Agents);
            DecideAction();
            // Clear the agent Decision subscription collection for the next update cycle.
            m_Agents.Clear();
        }

        /// <summary>
        /// Is called only once at the beginning of the training or inference session.
        /// </summary>
        protected abstract void Initialize();

        /// <summary>
        /// Is called once per Environment Step after the Brain has been initialized.
        /// </summary>
        protected abstract void DecideAction();
    }
}
