using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TeachingToggle : MonoBehaviour {

    bool collectData;
    Brain expertBrain;

	// Use this for initialization
	void Start () {
        collectData = true;
        expertBrain = GetComponent<Brain>();
	}
	
	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(KeyCode.B))
        {
            collectData = !collectData;
        }
        Monitor.Log("Collecting Data", collectData.ToString());
	}
}
