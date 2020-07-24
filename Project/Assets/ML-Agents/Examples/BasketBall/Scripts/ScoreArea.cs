using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScoreArea : MonoBehaviour
{
    public PlayerAgent agent;

    // Update is called once per frame
    void OnTriggerEnter(Collider c)
    {
        if (c.name == "Ball")
        {
        	// Debug.Log("ScoreArea Triggered");
        	agent.Hit();
        }
    }
}
