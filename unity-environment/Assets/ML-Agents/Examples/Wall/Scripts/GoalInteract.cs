using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalInteract : MonoBehaviour
{

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionEnter(Collision collision)
    {
        WallAgent agent = GameObject.Find("Agent").GetComponent<WallAgent>();
        agent.done = true;
        agent.reward = 1f;
	}

}