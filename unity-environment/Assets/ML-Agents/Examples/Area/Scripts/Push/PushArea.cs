using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushArea : Area {

    public GameObject block;
    public GameObject goalHolder;
    public GameObject academy;

	// Use this for initialization
	void Start () {
		academy = GameObject.Find("Academy");
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	public override void ResetArea()
	{
        float goalSize = academy.GetComponent<PushAcademy>().goalSize;
        float blockSize = academy.GetComponent<PushAcademy>().blockSize;
        float xVariation = academy.GetComponent<PushAcademy>().xVariation;

        block.transform.position = new Vector3(Random.Range(-xVariation, xVariation), 1f, -6f) + gameObject.transform.position;
        goalHolder.transform.position = new Vector3(Random.Range(-xVariation, xVariation), -0.1f, -2f) + gameObject.transform.position;
        goalSize = Random.Range(goalSize * 0.9f, goalSize * 1.1f);
        blockSize = Random.Range(blockSize * 0.9f, blockSize * 1.1f);
        block.transform.localScale = new Vector3(blockSize, 1f, blockSize);
        goalHolder.transform.localScale = new Vector3(goalSize, 1f, goalSize);
	}

}
