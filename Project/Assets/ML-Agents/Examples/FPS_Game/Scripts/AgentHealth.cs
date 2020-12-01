using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AgentHealth : MonoBehaviour
{
    public float currentHealth = 100;

    public float damagePerHit = 5;

    public Slider UISlider;
    // Start is called before the first frame update
    void OnEnable()
    {
        currentHealth = 100;
        UISlider.value = currentHealth;
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
            UISlider.value = currentHealth;
        }
    }
}
