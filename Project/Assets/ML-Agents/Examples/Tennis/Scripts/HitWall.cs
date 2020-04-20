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

    [HideInInspector]
    public FloorHit lastFloorHit;

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
        lastFloorHit = FloorHit.Service;
        m_LastAgentHit = -1;
    }

    void Reset()
    {
        m_AgentB.EndEpisode();
        m_AgentA.EndEpisode();
        //m_Area.MatchReset();
    }
    
    void AgentAWins()
    {
        m_AgentA.SetReward(1f);// + m_AgentA.energyPenalty);
        m_AgentB.SetReward(-1f);
        m_AgentA.score += 1;
        Reset();

    }

    void AgentBWins()
    {
        m_AgentA.SetReward(-1f);
        m_AgentB.SetReward(1f);// + m_AgentB.energyPenalty);
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
                if (m_LastAgentHit == 0 || lastFloorHit == FloorHit.FloorAHit)
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
                if (m_LastAgentHit == 1 || lastFloorHit == FloorHit.FloorBHit)
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
                if (m_LastAgentHit == 0 || lastFloorHit == FloorHit.FloorAHit || lastFloorHit == FloorHit.Service)
                {
                    AgentBWins();
                }
                else
                {
                    lastFloorHit = FloorHit.FloorAHit;
                }
            }
            else if (collision.gameObject.name == "floorB")
            {
                // Agent B hits into floor, double bounce or service
                if (m_LastAgentHit == 1 || lastFloorHit == FloorHit.FloorBHit || lastFloorHit == FloorHit.Service)
                {
                    AgentAWins();
                }
                else
                {
                    lastFloorHit = FloorHit.FloorBHit;
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
                lastFloorHit = FloorHit.FloorHitUnset;
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
                lastFloorHit = FloorHit.FloorHitUnset;
            }
        }
    }
}
