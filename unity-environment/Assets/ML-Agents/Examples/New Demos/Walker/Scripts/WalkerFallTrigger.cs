using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerFallTrigger : MonoBehaviour {

    public WalkerAgent agent;
    // WalkerAgentMotorJoints agent;

    void Start(){
        // agent = transform.root.gameObject.GetComponent<WalkerAgentMotorJoints>();
        agent = transform.root.gameObject.GetComponent<WalkerAgent>();
    }

    // private void OnCollisionEnter(Collision other)
    // {
    //     if (other.transform.CompareTag("ground"))
    //     {
    //         agent.Done();
    //         agent.AddReward(-1f);
    //     }

    // }
}
