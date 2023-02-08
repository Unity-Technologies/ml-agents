using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeFall : MonoBehaviour
{
    Rigidbody m_AgentRb;

    // Start is called before the first frame update
    void Start()
    {
        m_AgentRb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        //m_AgentRb.AddForce(transform.up * -0.2f, ForceMode.VelocityChange);   // This seems to slow things down.
        //transform.Rotate(transform.up, Time.deltaTime * 0f);    // This seems to slow things down.
    }
}
