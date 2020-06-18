using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmoothFollowObject : MonoBehaviour
{
    public Transform objectToFollow;
    public Vector3 offsetInObjectLocalSpace;
    public float smoothingTime = 2;

    private Vector3 velocity;

    // Update is called once per frame
    void Update()
    {
        transform.position = Vector3.SmoothDamp(transform.position,
            objectToFollow.transform.TransformPoint(offsetInObjectLocalSpace), ref velocity, smoothingTime);
    }
}
