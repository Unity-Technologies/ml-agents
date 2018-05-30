using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class contains logic for locomotion agents with joints which might make contact with the ground.
/// By attaching this as a component to those joints, their contact with the ground can be used as either
/// an observation for that agent, and/or a means of punishing the agent for making undesirable contact.
/// !!! You MUST tag the ground "ground", and the target "target" for this script to work
/// </summary>
public class CrawlerContact : MonoBehaviour
{
    [HideInInspector]
    public CrawlerAgent agent;
    public float contactPenalty;
    public bool touchingGround;
    public bool penalizeOnContact;
    private const string Ground = "ground"; 

    /// <summary>
    /// Obtain reference to agent.
    /// </summary>
    void Start()
    {
        agent = transform.root.GetComponent<CrawlerAgent>();
        agent = transform.parent.GetComponent<CrawlerAgent>();
        Physics.defaultSolverIterations = 6; //increasing this to increase solver accuracy
        Physics.defaultSolverVelocityIterations = 6; //increasing this to increase solver accuracy
    }

    /// <summary>
    /// Check for collision with ground, and optionally penalize agent.
    /// </summary>
    void OnCollisionEnter(Collision other)
    {
        // Touched the ground
        if (other.transform.CompareTag(Ground))
        {
            touchingGround = true;
            if (penalizeOnContact)
            {
                agent.AddReward(contactPenalty);
                agent.Done();
            }
        }

        // Touched the target 
        if (other.transform.CompareTag("target"))
        {
            agent.TouchedTarget(other.relativeVelocity.sqrMagnitude);
        }
    }

    /// <summary>
    /// Check for end of ground collision and reset flag appropriately.
    /// </summary>
    void OnCollisionExit(Collision other)
    {
        if (other.transform.CompareTag(Ground))
        {
            touchingGround = false;
        }
    }
}
