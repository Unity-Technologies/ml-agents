using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleNPC : MonoBehaviour
{

    public Transform target;

    private Rigidbody rb;

    public float walkSpeed = 1;

    private Vector3 dirToGo;
    // Start is called before the first frame update
    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        dirToGo = target.position - transform.position;
        dirToGo.y = 0;
        transform.rotation = Quaternion.LookRotation(dirToGo);
    }

    void FixedUpdate()
    {
        rb.AddForce(dirToGo.normalized * walkSpeed * Time.fixedDeltaTime, ForceMode.VelocityChange);
    }
}
