using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class BouncerBanana : MonoBehaviour {

    // Update is called once per frame
    void FixedUpdate () {
        gameObject.transform.Rotate(new Vector3(1, 0, 0), 0.5f);
    }

    private void OnTriggerEnter(Collider collision)
    {
        Agent agent = collision.gameObject.GetComponent<Agent>();
        if (agent != null)
        {
            agent.AddReward(1f);
            Respawn();
        }

    }

    public void Respawn(){
        gameObject.transform.localPosition = 
            new Vector3(
                (1 - 2 * Random.value) * 5f, 
                2f+ Random.value * 5f, 
                (1 - 2 * Random.value) * 5f);
    }

}
