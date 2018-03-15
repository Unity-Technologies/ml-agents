using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class hitWall : MonoBehaviour
{
    public GameObject areaObject;
    public int lastAgentHit;

    // Use this for initialization
    void Start()
    {
        lastAgentHit = -1;
    }

    private void OnTriggerExit(Collider other)
    {
        TennisArea area = areaObject.GetComponent<TennisArea>();
        TennisAgent agentA = area.agentA.GetComponent<TennisAgent>();
        TennisAgent agentB = area.agentB.GetComponent<TennisAgent>();

        if (other.name == "over")
        {
            if (lastAgentHit == 0)
            {
                agentA.AddReward( 0.1f);
            }
            else
            {
                agentB.AddReward(0.1f);
            }
            lastAgentHit = 0;

        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        TennisArea area = areaObject.GetComponent<TennisArea>();
        TennisAgent agentA = area.agentA.GetComponent<TennisAgent>();
        TennisAgent agentB = area.agentB.GetComponent<TennisAgent>();

        if (collision.gameObject.tag == "iWall")
        {
            if (collision.gameObject.name == "wallA")
            {
                if (lastAgentHit == 0)
                {
                    agentA.AddReward( -0.01f);
                    agentB.SetReward(0);
                    agentB.score += 1;
                }
                else
                {
                    agentA.SetReward(0);
                    agentB.AddReward(-0.01f);
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                if (lastAgentHit == 0)
                {
                    agentA.AddReward( -0.01f);
                    agentB.SetReward(0);
                    agentB.score += 1;
                }
                else
                {
                    agentA.SetReward(0);
                    agentB.AddReward( -0.01f);
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                if (lastAgentHit == 0 || lastAgentHit == -1)
                {
                    agentA.AddReward( -0.01f);
                    agentB.SetReward(0);
                    agentB.score += 1;
                }
                else
                {
                    agentA.AddReward( -0.01f);
                    agentB.SetReward(0);
                    agentB.score += 1;

                }
            }
            else if (collision.gameObject.name == "floorB")
            {
                if (lastAgentHit == 1 || lastAgentHit == -1)
                {
                    agentA.SetReward(0);
                    agentB.AddReward( -0.01f);
                    agentA.score += 1;
                }
                else
                {
                    agentA.SetReward(0);
                    agentB.AddReward( -0.01f);
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "net")
            {
                if (lastAgentHit == 0)
                {
                    agentA.AddReward( -0.01f);
                    agentB.SetReward(0);
                    agentB.score += 1;
                }
                else
                {
                    agentA.SetReward(0);
                    agentB.AddReward( -0.01f);
                    agentA.score += 1;
                }
            }
            agentA.Done();
            agentB.Done();
            area.MatchReset();
        }

        if (collision.gameObject.tag == "agent")
        {
            if (collision.gameObject.name == "AgentA")
            {
                lastAgentHit = 0;
            }
            else
            {
                lastAgentHit = 1;
            }
        }
    }
}