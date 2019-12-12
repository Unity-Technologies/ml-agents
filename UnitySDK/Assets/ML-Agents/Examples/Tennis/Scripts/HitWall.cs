using UnityEngine;

public class HitWall : MonoBehaviour
{
    public GameObject areaObject;
    public int lastAgentHit;
    public int lastFloorHit;

    TennisArea m_Area;
    TennisAgent m_AgentA;
    TennisAgent m_AgentB;

    // Use this for initialization
    void Start()
    {
        m_Area = areaObject.GetComponent<TennisArea>();
        m_AgentA = m_Area.agentA.GetComponent<TennisAgent>();
        m_AgentB = m_Area.agentB.GetComponent<TennisAgent>();
    }

    private void Reset()
    {
        m_AgentA.Done();
        m_AgentB.Done();
        m_Area.MatchReset();
        lastFloorHit = -1;
    }
    
    private void AgentAWins()
    {
        m_AgentA.SetReward(1);
        m_AgentB.SetReward(-1);
        m_AgentA.score += 1;
        Reset();

    }

    private void AgentBWins()
    {
        m_AgentA.SetReward(-1);
        m_AgentB.SetReward(1);
        m_AgentB.score += 1;
        Reset();

    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("iWall"))
        {
            if (collision.gameObject.name == "wallA")
            {
                //Agent A hits into wall or agent B hit a winner
                if (lastAgentHit == 0 || lastFloorHit == 0)
                {
                    AgentBWins();
                }
                //Agent B hits long
                else
                {
                    AgentAWins();
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                //Agent B hits into wall or agent A hit a winner
                if (lastAgentHit == 1 || lastFloorHit == 1)
                {
                    AgentAWins();
                }
                //Agent A hits long
                else
                {
                    AgentBWins();
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                //Agent A hits into floor or double bounce
                if (lastAgentHit == 0 || lastFloorHit == 0)
                {
                    AgentBWins();
                }
                else
                {
                    lastFloorHit = 0;
                }
            } 
            else if (collision.gameObject.name == "floorB")
            {
                //Agent B hits into floor or double bounce
                if (lastAgentHit == 1 || lastFloorHit == 1)
                {
                    AgentAWins();
                }
                else
                {
                    lastFloorHit = 1;
                }
            }
        }
        else if (collision.gameObject.name == "AgentA")
        {
            //Agent A double hit
            if (lastAgentHit == 0)
            {
                AgentBWins();
            }
            else
            {
                lastAgentHit = 0;
                lastFloorHit = -1;
            }
        }
        else if (collision.gameObject.name == "AgentB")
        {
            //Agent B double hit
            if (lastAgentHit == 1)
            {
                AgentAWins();
            }
            else
            {
                lastAgentHit = 1;
                lastFloorHit = -1;
            }
        }
    }
}
