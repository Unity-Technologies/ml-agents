using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerGroundContact : MonoBehaviour
{
    public int index;
    public WalkerAgent agent;
    public bool touchingGround;
    public bool penalizeOnContact;

    void Start()
    {
        agent = transform.root.GetComponent<WalkerAgent>();
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
