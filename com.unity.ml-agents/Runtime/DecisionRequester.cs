using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// The DecisionRequestor component automatically request decisions for an
    /// <see cref="Agent"/> instance at regular intervals.
    /// </summary>
    /// <remarks>
    /// Attach a DecisionRequestor component to the same [GameObject] as the
    /// <see cref="Agent"/> component.
    ///
    /// The DecisionRequestor component provides a convenient and flexible way to
    /// trigger the agent decision making process. Without a DecisionRequestor,
    /// your <see cref="Agent"/> implmentation must manually call its
    /// <seealso cref="Agent.RequestDecision"/> function.
    /// </remarks>
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
        /// </summary>
        /// <remarks>
        /// This does not affect <see cref="DecisionPeriod"/>. Turning this on will distribute
        /// the decision-making computations for all the agents across multiple Academy steps.
        /// This can be valuable in scenarios where you have many agents in the scene, particularly
        /// during the inference phase.
        /// </remarks>
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
