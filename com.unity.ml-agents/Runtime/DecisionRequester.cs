using System;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A component that when attached to an Agent will automatically request decisions from it
    /// at regular intervals.
    /// </summary>
    [AddComponentMenu("ML Agents/Decision Requester", (int)MenuGroup.Default)]
    public class DecisionRequester : MonoBehaviour
    {
        [Range(1, 20)]
        [Tooltip("The agent will automatically request a decision every X Academy steps.")]
        public int DecisionPeriod = 5;

        [Tooltip("Whether or not AgentAction will be called on Academy steps that decisions aren't requested. Has no effect if DecisionPeriod is 1.")]
        public bool RepeatAction = true;

        [Tooltip("Whether or not Agent decisions should start at a random offset.")]
        public bool offsetStep;

        Agent m_Agent;
        int m_Offset;
        public void Awake()
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
            if (RepeatAction)
            {
                m_Agent?.RequestAction();
            }
        }
    }
}
