using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BouncerBanana : MonoBehaviour {


    void Start(){

    }
	
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

    private void Respawn(){
        Vector3 oldPosition = gameObject.transform.position;
        gameObject.transform.position = new Vector3((1 - 2 * Random.value) * 15, oldPosition.y, (1 - 2 * Random.value) * 15);
    }

}
