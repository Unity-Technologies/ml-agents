using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentHealth : MonoBehaviour
{
    public float currentHealth = 100;

    public float damagePerHit = 5;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag("projectile"))
        {
            currentHealth -= damagePerHit;
        }
    }
}
