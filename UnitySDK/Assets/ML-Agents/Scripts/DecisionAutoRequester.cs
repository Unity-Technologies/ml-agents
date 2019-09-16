using UnityEngine;

namespace MLAgents
{
    [RequireComponent(typeof(Agent))]
    public class DecisionAutoRequester : MonoBehaviour
    {

        public int DecisionPeriod = 1;
        public bool RepeatAction = true;
        private Agent m_Agent;

        public void OnEnable()
        {
            DecisionPeriod = Mathf.Max(DecisionPeriod, 1);
            m_Agent = gameObject.GetComponent<Agent>();
            GameObject.FindObjectOfType<Academy>().AgentSetStatus += SetStatus;
        }

        void OnDisable()
        {
            GameObject.FindObjectOfType<Academy>().AgentSetStatus -= SetStatus;
        }

        void SetStatus(bool academyMaxStep, bool academyDone, int academyStepCounter)
        {
            if (academyStepCounter % DecisionPeriod == 0)
            {
                m_Agent.RequestDecision();
            }
            if (RepeatAction)
            {
                m_Agent.RequestAction();
            }
        }
    }
}
