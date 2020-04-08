using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// A component that when attached to an Agent will automatically request decisions from it
    /// at regular intervals.
    /// </summary>
    [AddComponentMenu("ML Agents/Decision Requester", (int)MenuGroup.Default)]
    [RequireComponent(typeof(Agent))]
    public class DecisionRequester : MonoBehaviour
    {
        /// <summary>
        /// The frequency with which the agent requests a decision. A DecisionPeriod of 5 means
        /// that the Agent will request a decision every 5 Academy steps. /// </summary>
        [Range(1, 20)]
        [Tooltip("The frequency with which the agent requests a decision. A DecisionPeriod " +
                 "of 5 means that the Agent will request a decision every 5 Academy steps.")]
        public int DecisionPeriod = 5;

        /// <summary>
        /// Indicates whether or not the agent will take an action during the Academy steps where
        /// it does not request a decision. Has no effect when DecisionPeriod is set to 1.
        /// </summary>
        [Tooltip("Indicates whether or not the agent will take an action during the Academy " +
                 "steps where it does not request a decision. Has no effect when DecisionPeriod " +
                 "is set to 1.")]
        [FormerlySerializedAs("RepeatAction")]
        public bool TakeActionsBetweenDecisions = true;

        [NonSerialized]
        Agent m_Agent;

        internal void Awake()
        {
            m_Agent = gameObject.GetComponent<Agent>();
            Debug.Assert(m_Agent != null, "Agent component was not found on this gameObject and is required.");
            Academy.Instance.AgentPreStep += MakeRequests;
        }

        void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentPreStep -= MakeRequests;
            }
        }

        /// <summary>
        /// Method that hooks into the Academy in order inform the Agent on whether or not it should request a
        /// decision, and whether or not it should take actions between decisions.
        /// </summary>
        /// <param name="academyStepCount">The current step count of the academy.</param>
        void MakeRequests(int academyStepCount)
        {
            if (academyStepCount % DecisionPeriod == 0)
            {
                m_Agent?.RequestDecision();
            }
            if (TakeActionsBetweenDecisions)
            {
                m_Agent?.RequestAction();
            }
        }
    }
}
