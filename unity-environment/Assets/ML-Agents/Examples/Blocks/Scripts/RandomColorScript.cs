using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomColorScript : MonoBehaviour {

    Renderer ren;

	// Use this for initialization
	void Start () {
        RandomColor();
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void RandomColor()
    {
        if (ren == null) ren = gameObject.GetComponent<Renderer>();
        ren.material.color = Random.ColorHSV(0, 1);
    }
}
