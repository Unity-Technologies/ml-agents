using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HitWall : MonoBehaviour
{
    public GameObject areaObject;
    public int lastAgentHit;

    private TennisArea area;
    private TennisAgent agentA;
    private TennisAgent agentB;

    private int nbPasses = 0;

    // Use this for initialization
    void Start()
    {
        area = areaObject.GetComponent<TennisArea>();
        agentA = area.agentA.GetComponent<TennisAgent>();
        agentB = area.agentB.GetComponent<TennisAgent>();
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.name == "over")
        {
            nbPasses +=1;
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
                    agentB.Score += 1;
                }
                else
                {
                    agentA.Score += 1;
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                if (lastAgentHit == 0)
                {
                    agentB.Score += 1;
                }
                else
                {
                    agentA.Score += 1;
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                agentB.Score += 1;
            }
            else if (collision.gameObject.name == "floorB")
            {
                agentA.Score += 1;
            }
            else if (collision.gameObject.name == "net")
            {
                if (lastAgentHit == 0)
                {
                    agentB.Score += 1;
                }
                else
                {
                    agentA.Score += 1;
                }
            }
            agentA.Done();
            agentB.Done();
            area.MatchReset();
            TennisArea.AddPasses(nbPasses);
            nbPasses = 0;
        }

        if (collision.gameObject.CompareTag("agent"))
        {
            lastAgentHit = collision.gameObject.name == "AgentA" ? 0 : 1;
        }
    }
}