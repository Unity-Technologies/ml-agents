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
        public int DecisionPeriod;
        public bool RepeatAction = true;
        private Agent m_Agent;
        private int m_Counter;
        void Awake()
        {
            m_Agent = gameObject.GetComponent<Agent>();
        }
        void FixedUpdate()
        {
            if (m_Counter % DecisionPeriod == 0)
            {
                m_Agent?.RequestDecision();
            }
            if (RepeatAction)
            {
                m_Agent?.RequestAction();
            }
            m_Counter++;
        }

    }
}