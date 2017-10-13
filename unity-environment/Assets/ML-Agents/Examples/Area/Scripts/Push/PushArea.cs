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
		block.transform.position = new Vector3(Random.Range(-2.5f, 2.5f), 1f, Random.Range(-7f, -5f)) + gameObject.transform.position;
        goalHolder.transform.position = new Vector3(Random.Range(-3.5f, 3.5f), -0.1f, Random.Range(0f, -3f)) + gameObject.transform.position;

        float size = academy.GetComponent<PushAcademy>().objectSize;

        block.transform.localScale = new Vector3(size, 1f, size);
        goalHolder.transform.localScale = new Vector3(size + 1f, 1f, size + 1f);
	}

}
