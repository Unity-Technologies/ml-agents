using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class contains logic for locomotion agents with joints which might make contact with the ground.
/// By attaching this as a component to those joints, their contact with the ground can be used as either
/// an observation for that agent, and/or a means of punishing the agent for making undesirable contact.
/// </summary>
public class GroundContact : MonoBehaviour
{
    public int index;
    public Agent agent;
    public bool touchingGround;
    public bool penalizeOnContact;

    void Start()
    {
        agent = transform.root.GetComponent<Agent>();
    }

    void OnCollisionEnter(Collision other)
    {
        if (other.transform.CompareTag("ground"))
        {
            touchingGround = true;
            if (penalizeOnContact)
            {
                agent.Done();
                agent.SetReward(-1f);
            }
        }
    }

    void OnCollisionExit(Collision other)
    {
        if (other.transform.CompareTag("ground"))
        {
            touchingGround = false;
        }
    }
}
