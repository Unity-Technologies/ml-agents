using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DistanceScaling : MonoBehaviour {

    public GameObject renderObject;

    public float minDistance, maxDistance;

    public float scalingAmount;

    private float initialSize;

    private float normalizedDistance;

    private float scaledAmount;

    private float scaledSize;

    private float dist;

	// Use this for initialization
	void Start () {

        initialSize = renderObject.GetComponent<Renderer>().material.GetFloat("_LineSize");


	}
	
	// Update is called once per frame
	void Update () {

        dist = Vector3.Distance(this.transform.position, renderObject.transform.position);

        if (dist < minDistance) {
            normalizedDistance = 0;
        }
        else
        {
            normalizedDistance = dist / maxDistance;
        }

        scaledAmount = scalingAmount * normalizedDistance;

        scaledSize = (initialSize + scaledAmount);

        renderObject.GetComponent<Renderer>().material.SetFloat("_LineSize", scaledSize);
        

	}
}
