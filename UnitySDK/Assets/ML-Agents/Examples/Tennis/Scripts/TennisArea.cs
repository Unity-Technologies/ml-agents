using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TennisArea : MonoBehaviour {

    public GameObject ball;
    public GameObject agentA;
    public GameObject agentB;
    private Rigidbody ballRb;

    // Use this for initialization
    void Start ()
    {
        ballRb = ball.GetComponent<Rigidbody>();
        MatchReset();
    }
    
    public void MatchReset() 
    {
        float ballOut = Random.Range(6f, 8f);
        int flip = Random.Range(0, 2);
        if (flip == 0)
        {
            ball.transform.position = new Vector3(-ballOut, 6f, 0f) + transform.position;
        }
        else
        {
            ball.transform.position = new Vector3(ballOut, 6f, 0f) + transform.position;
        }
        ballRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.localScale = new Vector3(1, 1, 1);
        ball.GetComponent<HitWall>().lastAgentHit = -1;
    }

    void FixedUpdate() 
    {
        Vector3 rgV = ballRb.velocity;
        ballRb.velocity = new Vector3(Mathf.Clamp(rgV.x, -9f, 9f), Mathf.Clamp(rgV.y, -9f, 9f), rgV.z);
    }
}
