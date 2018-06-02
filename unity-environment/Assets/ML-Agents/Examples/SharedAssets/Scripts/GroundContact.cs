using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class contains logic for locomotion agents with joints which might make contact with the ground or target.
/// By attaching this as a component to those joints, their contact with the ground can be used as either
/// an observation for that agent, and/or a means of punishing the agent for making undesirable contact.
/// </summary>
public class GroundContact : MonoBehaviour
{
    public Agent agent;

    [Header("Ground Check")]
    // public bool checkForGroundContact = true; //check for ground contact?
    public bool agentDoneOnGroundContact = true; //reset agent on ground contact?
    public bool penalizeGroundContact; //if this body part touches the ground should the agent be penalized?
    public float groundContactPenalty; //penalty amount (ex: -1)
    public bool touchingGround;
    private const string Ground = "ground"; //tag on ground obj


    // [Header("Detect Targets")]
    // [Space(10)]
    // public bool checkForTargetContact = false; //check for target contact?
    // // public bool resetAgentOnTargetContact = true; //reset agent on target contact?
    // // public bool rewardTargetContact; //if this body part touches the target should the agent be reward?
    // // public float touchedTargetReward; //reward amount (ex: 1)
    // public bool touchingTarget;
    // private const string Target = "target"; //tag on target obj

    // enum AgentType
    // {
    //     walker, crawler, dog
    // }
    // AgentType agentType = new AgentType();

    /// <summary>
    /// Obtain reference to agent.
    /// </summary>
    void Awake()
    {
        // agent = transform.root.GetComponent<Agent>();

    }

    /// <summary>
    /// Check for collision with ground, and optionally penalize agent.
    /// </summary>
    void OnCollisionEnter(Collision col)
    {
        // if(checkForGroundContact)
        // {
            if (col.transform.CompareTag(Ground))
            {
                touchingGround = true;
                // print("touching ground");
                if (penalizeGroundContact)
                {
                    agent.SetReward(groundContactPenalty);
                }
                if (agentDoneOnGroundContact)
                {
                    agent.Done();
                }
            // }
        }

        // if(checkForTargetContact)
        // {
        //     // Touched the target 
        //     if (col.transform.CompareTag(Target))
        //     {
        //         touchingTarget = true;
        //         // if (rewardTargetContact)
        //         // {
        //         //     agent.SetReward(touchedTargetReward);
        //         // }
        //         // if(resetAgentOnTargetContact)
        //         // {
        //         //     agent.Done();
        //         // }
        //     }
        // }
    }

    /// <summary>
    /// Check for end of ground collision and reset flag appropriately.
    /// </summary>
    void OnCollisionExit(Collision other)
    {
        // if(checkForGroundContact)
        // {
            if (other.transform.CompareTag(Ground))
            {
                touchingGround = false;
            }
        // }
        // if(checkForTargetContact)
        // {
        //     if (other.transform.CompareTag(Target))
        //     {
        //         touchingTarget = false;
        //     }
        // }
    }
}
