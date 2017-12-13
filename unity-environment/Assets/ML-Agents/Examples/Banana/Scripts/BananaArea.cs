using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaArea : Area {

    public GameObject banana;
    public int numBananas;
    public bool respawnBananas;

	// Use this for initialization
	void Start () {
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	public override void ResetArea()
	{
        float range = 45f;

        GameObject[] oldBanans = GameObject.FindGameObjectsWithTag("banana");
        foreach (GameObject bana in oldBanans) {
            Destroy(bana);
        }

        GameObject[] agents = GameObject.FindGameObjectsWithTag("agent");
        foreach (GameObject agent in agents)
        {
            agent.transform.position = new Vector3(Random.Range(-range, range), 2f, Random.Range(-range, range)) + transform.position;
            agent.transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
        }

        for (int i = 0; i < numBananas; i++) {
            GameObject bana = Instantiate(banana, new Vector3(Random.Range(-range, range), 2f, Random.Range(-range, range)) + transform.position, banana.gameObject.transform.rotation);
            bana.GetComponent<BananaLogic>().respawn = true;
        }
	}

}
