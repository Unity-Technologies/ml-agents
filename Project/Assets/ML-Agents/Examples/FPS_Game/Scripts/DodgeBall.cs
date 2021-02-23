using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DodgeBall : MonoBehaviour
{

    public bool inPlay;

    [HideInInspector]
    public Rigidbody rb;

    public Collider BallCollider;

    public int TeamToIgnore;
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

    //Set ball to either a pickup item or an active ball that is in play
    //inPlay = true means the ball can hurt other players
    //ignoreTeam = the TeamID to ignore. *friendly fire mechanic
    public void BallIsInPlay(bool p, int ignoreTeam = -1)
    {
        if (p)
        {
            TagBallAs( "dodgeBallActive");
        }
        else
        {
            TagBallAs( "dodgeBallPickup");
        }
        inPlay = p;
        TeamToIgnore = ignoreTeam;
    }

    void TagBallAs(string tag)
    {
        gameObject.tag = tag;
        BallCollider.gameObject.tag = tag;
    }

    private void OnCollisionEnter(Collision col)
    {
        //IF NOT MY TEAM
        //PLAYER GOES TO TIMEOUT
        if (col.gameObject.CompareTag("ground"))
        {
            BallIsInPlay(false);
        }
    }
}
