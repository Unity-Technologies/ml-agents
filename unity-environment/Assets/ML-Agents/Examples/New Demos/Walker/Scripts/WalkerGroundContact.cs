using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerGroundContact : MonoBehaviour {

    public int index;
    public WalkerAgent agent;
    public bool touchingGround;
    public bool penalizeOnContact;
    // public Rigidbody rb;
    // WalkerAgentMotorJoints agent;

    void Start(){
        // agent = transform.root.GetComponent<WalkerAgentMotorJoints>();
        agent = transform.root.GetComponent<WalkerAgent>();
        // rb = GetComponent<Rigidbody>();
    }

    // void OnCollisionStay(Collision other){
    //     if (other.transform.CompareTag("ground"))
    //     {
    //         agent.leg_touching[index] = true;
    //     }
    // }
    void OnCollisionEnter(Collision other)
    {
        if (other.transform.CompareTag("ground"))
        {
            // agent.bodyParts[transform].g
            // agent.leg_touching[index] = true;
            touchingGround = true;
            if(penalizeOnContact)
            {
                agent.Done();
                agent.AddReward(-1f);
            }
        }
    }
    void OnCollisionExit(Collision other){
        if (other.transform.CompareTag("ground"))
        {
            // agent.leg_touching[index] = false;
            touchingGround = false;
        }
    }

}
