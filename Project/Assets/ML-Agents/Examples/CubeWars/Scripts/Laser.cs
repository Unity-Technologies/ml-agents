using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Laser : MonoBehaviour
{
    public bool isFired;
    public float maxLength = 25f;
    public float width = 1f;
    public bool animate = false;
    LineRenderer laserRenderer;
    // Start is called before the first frame update
    void Start()
    {
        laserRenderer = GetComponentInChildren<LineRenderer>();
        laserRenderer.material.SetTextureScale("_MainTex", new Vector2(0.07f, 1.0f));
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        laserRenderer.SetWidth(width, width);
        if (isFired)
        {
            if (animate)
            {
                laserRenderer.material.SetTextureOffset("_MainTex", new Vector2(-6 * Time.time, 0.0f));
            }
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, width, transform.forward, out hit, maxLength))
            {
                float hitLength = Vector3.Distance(hit.point, transform.position);
                laserRenderer.SetPosition(1, new Vector3(0f, 0f, hitLength / transform.lossyScale.z));
            }
            else
            {
                laserRenderer.SetPosition(1, new Vector3(0f, 0f, maxLength / transform.lossyScale.z));
            }
        }
        else
        {
            laserRenderer.SetPosition(1, new Vector3(0f, 0f, 0f));
        }

    }
}
