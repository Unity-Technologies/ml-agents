using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Laser : MonoBehaviour
{
    public bool isFired;
    public float maxLength = 25f;
    public float width = 0.5f;
    LineRenderer laserRenderer;
    // Start is called before the first frame update
    void Start()
    {
        laserRenderer = GetComponentInChildren<LineRenderer>();
        laserRenderer.SetWidth(width, width);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if(isFired)
        {
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.forward, out hit, maxLength))
            {
                laserRenderer.SetPosition(1, new Vector3(0f, 0f, hit.distance/transform.lossyScale.z));
            }
            else
            {
                laserRenderer.SetPosition(1, new Vector3(0f, 0f, maxLength/transform.lossyScale.z));
            }
        }
        else
        {
            laserRenderer.SetPosition(1, new Vector3(0f, 0f, 0f));
        }

    }
}
