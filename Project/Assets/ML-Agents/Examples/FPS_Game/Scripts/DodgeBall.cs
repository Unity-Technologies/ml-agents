using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DodgeBall : MonoBehaviour
{

    public bool inPlay;

    [HideInInspector]
    public Rigidbody rb;
    // Start is called before the first frame update
    void Start()
    {

    }


    private void OnEnable()
    {
        // inPlay = false;
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionEnter(Collision col)
    {
        //IF NOT MY TEAM
        //PLAYER GOES TO TIMEOUT
        if (col.gameObject.CompareTag("ground"))
        {
            inPlay = false;
        }
    }
}
