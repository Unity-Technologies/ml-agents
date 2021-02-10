using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DodgeBall : MonoBehaviour
{

    public bool inPlay;

    // Start is called before the first frame update
    void Start()
    {

    }


    private void OnEnable()
    {
        inPlay = true;
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionEnter(Collision other)
    {
        //IF NOT MY TEAM
        //PLAYER GOES TO TIMEOUT
    }
}
