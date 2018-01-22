using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaLogic : MonoBehaviour {

    public bool respawn;

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void OnEaten() {
        if (respawn) {
            transform.position = new Vector3(Random.Range(-45f, 45f), transform.position.y + 3f, Random.Range(-45f, 45f));
        }
        else {
            Destroy(gameObject);
        }
    }
}
