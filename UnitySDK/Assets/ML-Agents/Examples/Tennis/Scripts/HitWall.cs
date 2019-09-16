using UnityEngine;

public class HitWall : MonoBehaviour
{
    public GameObject areaObject;
    public int lastAgentHit;

    private TennisArea m_Area;
    private TennisAgent m_AgentA;
    private TennisAgent m_AgentB;

    // Use this for initialization
    void Start()
    {
        m_Area = areaObject.GetComponent<TennisArea>();
        m_AgentA = m_Area.agentA.GetComponent<TennisAgent>();
        m_AgentB = m_Area.agentB.GetComponent<TennisAgent>();
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.name == "over")
        {
            if (lastAgentHit == 0)
            {
                m_AgentA.AddReward(0.1f);
            }
            else
            {
                m_AgentB.AddReward(0.1f);
            }
            lastAgentHit = 0;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("iWall"))
        {
            if (collision.gameObject.name == "wallA")
            {
                if (lastAgentHit == 0)
                {
                    m_AgentA.AddReward(-0.01f);
                    m_AgentB.SetReward(0);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(0);
                    m_AgentB.AddReward(-0.01f);
                    m_AgentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                if (lastAgentHit == 0)
                {
                    m_AgentA.AddReward(-0.01f);
                    m_AgentB.SetReward(0);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(0);
                    m_AgentB.AddReward(-0.01f);
                    m_AgentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                if (lastAgentHit == 0 || lastAgentHit == -1)
                {
                    m_AgentA.AddReward(-0.01f);
                    m_AgentB.SetReward(0);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.AddReward(-0.01f);
                    m_AgentB.SetReward(0);
                    m_AgentB.score += 1;
                }
            }
            else if (collision.gameObject.name == "floorB")
            {
                if (lastAgentHit == 1 || lastAgentHit == -1)
                {
                    m_AgentA.SetReward(0);
                    m_AgentB.AddReward(-0.01f);
                    m_AgentA.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(0);
                    m_AgentB.AddReward(-0.01f);
                    m_AgentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "net")
            {
                if (lastAgentHit == 0)
                {
                    m_AgentA.AddReward(-0.01f);
                    m_AgentB.SetReward(0);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(0);
                    m_AgentB.AddReward(-0.01f);
                    m_AgentA.score += 1;
                }
            }
            m_AgentA.Done();
            m_AgentB.Done();
            m_Area.MatchReset();
        }

        if (collision.gameObject.CompareTag("agent"))
        {
            lastAgentHit = collision.gameObject.name == "AgentA" ? 0 : 1;
        }
    }
}
