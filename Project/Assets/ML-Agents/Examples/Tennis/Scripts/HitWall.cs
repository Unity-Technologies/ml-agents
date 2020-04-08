using UnityEngine;

public class HitWall : MonoBehaviour
{
    public GameObject areaObject;
    int m_LastAgentHit;

    public enum FloorHit
        {
            Service,
            FloorHitUnset,
            FloorAHit,
            FloorBHit
        }

    FloorHit m_LastFloorHit;

    TennisArea m_Area;
    TennisAgent m_AgentA;
    TennisAgent m_AgentB;

    //  Use this for initialization
    void Start()
    {
        m_Area = areaObject.GetComponent<TennisArea>();
        m_AgentA = m_Area.agentA.GetComponent<TennisAgent>();
        m_AgentB = m_Area.agentB.GetComponent<TennisAgent>();
    }

    public void ResetPoint()
    {
        m_LastFloorHit = FloorHit.Service;
        m_LastAgentHit = -1;
    }

    void Reset()
    {
        m_AgentA.EndEpisode();
        m_AgentB.EndEpisode();
        m_Area.MatchReset();
    }
    
    void AgentAWins()
    {
        m_AgentA.SetReward(1 + m_AgentA.timePenalty);
        m_AgentB.SetReward(-1);
        m_AgentA.score += 1;
        Reset();

    }

    void AgentBWins()
    {
        m_AgentA.SetReward(-1);
        m_AgentB.SetReward(1 + m_AgentB.timePenalty);
        m_AgentB.score += 1;
        Reset();

    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("iWall"))
        {
            if (collision.gameObject.name == "wallA")
            {
                // Agent A hits into wall or agent B hit a winner
                if (m_LastAgentHit == 0 || m_LastFloorHit == FloorHit.FloorAHit)
                {
                    AgentBWins();
                }
                // Agent B hits long
                else
                {
                    AgentAWins();
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                // Agent B hits into wall or agent A hit a winner
                if (m_LastAgentHit == 1 || m_LastFloorHit == FloorHit.FloorBHit)
                {
                    AgentAWins();
                }
                // Agent A hits long
                else
                {
                    AgentBWins();
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                // Agent A hits into floor, double bounce or service
                if (m_LastAgentHit == 0 || m_LastFloorHit == FloorHit.FloorAHit || m_LastFloorHit == FloorHit.Service)
                {
                    AgentBWins();
                }
                else
                {
                    m_LastFloorHit = FloorHit.FloorAHit;
                }
            }
            else if (collision.gameObject.name == "floorB")
            {
                // Agent B hits into floor, double bounce or service
                if (m_LastAgentHit == 1 || m_LastFloorHit == FloorHit.FloorBHit || m_LastFloorHit == FloorHit.Service)
                {
                    AgentAWins();
                }
                else
                {
                    m_LastFloorHit = FloorHit.FloorBHit;
                }
            }
        }
        else if (collision.gameObject.name == "AgentA")
        {
            // Agent A double hit
            if (m_LastAgentHit == 0)
            {
                AgentBWins();
            }
            else
            {

                m_LastAgentHit = 0;
                m_LastFloorHit = FloorHit.FloorHitUnset;
            }
        }
        else if (collision.gameObject.name == "AgentB")
        {
            // Agent B double hit
            if (m_LastAgentHit == 1)
            {
                AgentAWins();
            }
            else
            {
                m_LastAgentHit = 1;
                m_LastFloorHit = FloorHit.FloorHitUnset;
            }
        }
    }
}
