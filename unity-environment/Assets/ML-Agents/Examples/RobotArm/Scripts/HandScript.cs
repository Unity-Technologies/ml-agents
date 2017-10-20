using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandScript : MonoBehaviour {

    public Vector3 OutHandPosition;

	// Update is called once per frame
	void Update () {
        OutHandPosition = transform.position;
    }
}
