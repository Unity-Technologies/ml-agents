using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReacherGoal : MonoBehaviour {

    public GameObject agent;
    public GameObject hand;
    public GameObject goalOn;

    // Use this for initialization
    void Start () {
        
    }
    
    // Update is called once per frame
    void Update () {
        
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == hand)
        {
            goalOn.transform.localScale = new Vector3(1f, 1f, 1f);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject == hand)
        {
            goalOn.transform.localScale = new Vector3(0f, 0f, 0f);
        }
    }

    private void OnTriggerStay(Collider other) 
    {
        if (other.gameObject == hand)
        {
            agent.GetComponent<ReacherAgent>().AddReward(0.01f);
        }
    }
}
