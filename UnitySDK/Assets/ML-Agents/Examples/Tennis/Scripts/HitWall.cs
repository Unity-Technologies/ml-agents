using UnityEngine;

public class HitWall : MonoBehaviour
{
    public GameObject areaObject;
    public int lastAgentHit;
    public int lastFloorHit;

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

    //private void OnTriggerExit(Collider other)
    //{
    //    if (other.name == "over")
    //    {
    //        if (lastAgentHit == 0)
    //        {
    //            m_AgentA.AddReward(0.1f);
    //        }
    //        else
    //        {
    //            m_AgentB.AddReward(0.1f);
    //        }
    //        lastAgentHit = 0;
    //    }
    //}
    private void reset()
    {
        m_AgentA.Done();
        m_AgentB.Done();
        m_Area.MatchReset();
        lastFloorHit = -1;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("iWall"))
        {
            if (collision.gameObject.name == "wallA")
            {
                if (lastAgentHit == 0 || lastFloorHit == 0)
                {
                    m_AgentA.SetReward(-1);
                    m_AgentB.SetReward(1);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(1);
                    m_AgentB.SetReward(-1);
                    m_AgentA.score += 1;
                }
                reset();
            }
            else if (collision.gameObject.name == "wallB")
            {
                if (lastAgentHit == 1 || lastFloorHit == 1)
                {
                    m_AgentA.SetReward(1);
                    m_AgentB.SetReward(-1);
                    m_AgentB.score += 1;
                }
                else
                {
                    m_AgentA.SetReward(-1);
                    m_AgentB.SetReward(1);
                    m_AgentA.score += 1;
                }
                reset();
            }
            else if (collision.gameObject.name == "floorA")
            {
                if (lastAgentHit == 0 || lastFloorHit == 0)
                {
                    m_AgentA.SetReward(-1);
                    m_AgentB.SetReward(1);
                    m_AgentB.score += 1;
                    reset();
                }
                else
                {
                    lastFloorHit = 0;
                }
            } 
            else if (collision.gameObject.name == "floorB")
            {
                if (lastAgentHit == 1 || lastFloorHit == 1)
                {
                    m_AgentA.SetReward(1);
                    m_AgentB.SetReward(-1);
                    m_AgentB.score += 1;
                    reset();
                }
                else
                {
                    lastFloorHit = 1;
                }
            }
       }

        if (collision.gameObject.CompareTag("agent"))
        {
            lastAgentHit = collision.gameObject.name == "AgentA" ? 0 : 1;
            lastFloorHit = -1;
        }
    }
}
