using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{

    /// <summary>
    /// This class contains logic for locomotion agents with joints which might make contact with the ground.
    /// By attaching this as a component to those joints, their contact with the ground can be used as either
    /// an observation for that agent, and/or a means of punishing the agent for making undesirable contact.
    /// </summary>
    public class GroundContact : MonoBehaviour
    {
        public Agent agent;
        public float contactPenalty;
        public bool touchingGround;
        public bool penalizeOnContact;
        private const string Ground = "ground";

        /// <summary>
        /// Obtain reference to agent.
        /// </summary>
        void Start()
        {
            agent = transform.root.GetComponent<Agent>();
        }

        /// <summary>
        /// Check for collision with ground, and optionally penalize agent.
        /// </summary>
        void OnCollisionEnter(Collision other)
        {
            if (other.transform.CompareTag(Ground))
            {
                touchingGround = true;
                if (penalizeOnContact)
                {
                    agent.Done();
                    agent.SetReward(contactPenalty);
                }
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
}
