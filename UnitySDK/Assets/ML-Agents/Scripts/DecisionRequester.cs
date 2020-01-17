using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using MLAgents.Sensor;
using UnityEngine.Serialization;

namespace MLAgents
{
    public class DecisionRequester : MonoBehaviour
    {
        public int DecisionPeriod = 5;
        public bool RepeatAction = true;
        private Agent m_Agent;
        public void Awake()
        {
            m_Agent = gameObject.GetComponent<Agent>();
            Academy.Instance.AgentSetStatus += MakeRequests;
        }
        void OnDestroy()
        {
            Academy.Instance.AgentSetStatus -= MakeRequests;
        }
        void MakeRequests(int count)
        {
            if (count % DecisionPeriod == 0)
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