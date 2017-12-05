using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class hitWall : MonoBehaviour
{
    public GameObject areaObject;
    int lastAgentHit;

    // Use this for initialization
    void Start()
    {
        lastAgentHit = -1;
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
                    agentA.reward = -0.1f;
                    agentB.reward = 0;
                    agentB.score += 1;
                }
                else
                {
                    agentA.reward = 0;
                    agentB.reward = -0.1f;
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "wallB")
            {
                if (lastAgentHit == 0)
                {
                    agentA.reward = -0.1f;
                    agentB.reward = 0;
                    agentB.score += 1;
                }
                else
                {
                    agentA.reward = 0;
                    agentB.reward = -0.1f;
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "floorA")
            {
                if (lastAgentHit != 1)
                {
                    agentA.reward = -0.1f;
                    agentB.reward = 0;
                    agentB.score += 1;
                }
                else
                {
                    agentA.reward = -0.1f;
                    agentB.reward = 0.1f;
                    agentB.score += 1;

                }
            }
            else if (collision.gameObject.name == "floorB")
            {
                if (lastAgentHit == 0)
                {
                    agentA.reward = 0.1f;
                    agentB.reward = -0.1f;
                    agentA.score += 1;
                }
                else
                {
                    agentA.reward = 0;
                    agentB.reward = -0.1f;
                    agentA.score += 1;
                }
            }
            else if (collision.gameObject.name == "net")
            {
                if (lastAgentHit == 0)
                {
                    agentA.reward = -0.1f;
                    agentB.reward = 0.0f;
                    agentB.score += 1;
                }
                else
                {
                    agentA.reward = 0.0f;
                    agentB.reward = -0.1f;
                    agentA.score += 1;
                }
            }
            area.MatchReset();
            agentA.done = true;
            agentB.done = true;
        }

        if (collision.gameObject.tag == "agent")
        {
            if (collision.gameObject.name == "AgentA")
            {
                if (lastAgentHit != 0)
                {
                    agentA.reward += 0.1f;
                    agentB.reward += 0.05f;
                }
                else 
                {
                    agentA.reward += 0.01f;
                }
                lastAgentHit = 0;
            }
            else
            {
                if (lastAgentHit != 1)
                {
                    agentB.reward += 0.1f;
                    agentA.reward += 0.05f;
                }
                else
                {
                    agentB.reward += 0.01f;
                }
                lastAgentHit = 1;
            }
        }
    }
}
