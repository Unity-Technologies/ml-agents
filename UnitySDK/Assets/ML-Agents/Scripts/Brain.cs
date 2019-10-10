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

        [NonSerialized]
        private bool m_IsInitialized;

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
        protected void LazyInitialize()
        {
            if (!m_IsInitialized)
            {
                var academy = FindObjectOfType<Academy>();
                if (academy)
                {
                    m_IsInitialized = true;
                    academy.BrainDecideAction += BrainDecideAction;
                    academy.DestroyAction += Shutdown;
                    Initialize();
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
