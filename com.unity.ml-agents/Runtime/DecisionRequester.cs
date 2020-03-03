using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// A component that when attached to an Agent will automatically request decisions from it
    /// at regular intervals.
    /// </summary>
    [AddComponentMenu("ML Agents/Decision Requester", (int)MenuGroup.Default)]
    internal class DecisionRequester : MonoBehaviour
    {
        /// <summary>
        /// The frequency with which the agent requests a decision. A DecisionPeriod of 5 means
        /// that the Agent will request a decision every 5 Academy steps.
        /// </summary>
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

        /// <summary>
        /// Whether or not the Agent decisions should start at an offset (different for each agent).
        /// This does not affect <see cref="DecisionPeriod"/>. Turning this on will distribute
        /// the decision-making computations for all the agents across multiple Academy steps.
        /// This can be valuable in scenarios where you have many agents in the scene, particularly
        /// during the inference phase.
        /// </summary>
        [Tooltip("Whether or not Agent decisions should start at an offset.")]
        public bool offsetStep;

        Agent m_Agent;
        int m_Offset;

        internal void Awake()
        {
            m_Offset = offsetStep ? gameObject.GetInstanceID() : 0;
            m_Agent = gameObject.GetComponent<Agent>();
            Academy.Instance.AgentSetStatus += MakeRequests;
        }

        void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentSetStatus -= MakeRequests;
            }
        }

        void MakeRequests(int count)
        {
            if ((count + m_Offset) % DecisionPeriod == 0)
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
