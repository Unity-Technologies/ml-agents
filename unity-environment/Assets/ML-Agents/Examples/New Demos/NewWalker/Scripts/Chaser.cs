using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Chaser : MonoBehaviour
{
    public Transform target;
    public float speed;
    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Vector3 targetDir = (target.transform.position - transform.position).normalized;
        GetComponent<Rigidbody>().AddForce(targetDir * speed, ForceMode.Acceleration);
    }
}
