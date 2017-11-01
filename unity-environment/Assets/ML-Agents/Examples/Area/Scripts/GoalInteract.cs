using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalInteract : MonoBehaviour
{
    public GameObject myAgent;
    public GameObject myObject;
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
        if (collision.gameObject == myObject)
        {
            Agent agent = myAgent.GetComponent<Agent>();
            agent.done = true;
            agent.reward = 1f;
        }
	}

}