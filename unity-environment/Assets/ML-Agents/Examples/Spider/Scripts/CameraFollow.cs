using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour {

	public Transform target;
	Vector3 offset;

	// Use this for initialization
	void Start () {
		offset = gameObject.transform.position - target.position;
	}
	
	// Update is called once per frame
	void Update () {
		gameObject.transform.position = target.position + offset;
	}
}
