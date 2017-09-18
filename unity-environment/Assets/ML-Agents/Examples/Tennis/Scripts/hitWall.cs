using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class hitWall : MonoBehaviour
{

    int lastAgentHit;

    // Use this for initialization
    void Start()
    {
        lastAgentHit = -1;
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionEnter(Collision collision)
    {
        TennisAgent agentA = GameObject.Find("AgentA").GetComponent<TennisAgent>();
        TennisAgent agentB = GameObject.Find("AgentB").GetComponent<TennisAgent>();
        TennisAcademy academy = GameObject.Find("Academy").GetComponent<TennisAcademy>();

        if (collision.gameObject.tag == "iWall")
        {
            academy.done = true;
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
        }

        if (collision.gameObject.tag == "agent")
        {
            if (collision.gameObject.name == "AgentA")
            {
                if (lastAgentHit != 0)
                {
                    agentA.reward = 0.1f;
                    agentB.reward = 0.05f;
                }
                lastAgentHit = 0;
            }
            else
            {
                if (lastAgentHit != 1)
                {
                    agentB.reward = 0.1f;
                    agentA.reward = 0.05f;
                }
                lastAgentHit = 1;
            }
        }
    }
}
