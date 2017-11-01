using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneEngine : MonoBehaviour {

    public float maxPower;
    [HideInInspector]
    public float powerMultiplier;

    Rigidbody rb;
    Transform tr;

    void Start(){
        rb = gameObject.GetComponent<Rigidbody>();
        tr = gameObject.transform;
    }

    void FixedUpdate(){
        powerMultiplier = Mathf.Max(-1f, Mathf.Min(powerMultiplier, 1f));
        rb.AddForce(tr.up * maxPower * powerMultiplier);
    }
}
