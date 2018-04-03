using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TennisAgent : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public bool invertX;
    public float invertMult;
    public int score;
    public GameObject scoreText;
    public GameObject myArea;
    public GameObject opponent;
    private Rigidbody rb;
    private Rigidbody ballRb;

    public override void InitializeAgent()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        ballRb = ball.GetComponent<Rigidbody>();
    }

    public override void CollectObservations()
    {
        AddVectorObs(invertMult * (gameObject.transform.position.x - myArea.transform.position.x));
        AddVectorObs(gameObject.transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * rb.velocity.x);
        AddVectorObs(rb.velocity.y);

        AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * ballRb.velocity.x);
        AddVectorObs(ballRb.velocity.y);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        float moveX = 0.25f * ScaleContinuousAction(vectorAction[0], -2f, 2f) * invertMult;
        
        if (ScaleContinuousAction(vectorAction[1], -2f, 2f) > 0f && gameObject.transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            float moveY = 0.5f;
            rb.velocity = new Vector3(rb.velocity.x, moveY * 12f, 0f);
        }

        rb.velocity = new Vector3(moveX * 50f, rb.velocity.y, 0f);

        if ((invertX && gameObject.transform.position.x - transform.parent.transform.position.x < -invertMult) ||
            (!invertX && gameObject.transform.position.x - transform.parent.transform.position.x > -invertMult))
        {
                gameObject.transform.position = new Vector3(-invertMult + transform.parent.transform.position.x, 
                                                            gameObject.transform.position.y, 
                                                            gameObject.transform.position.z);
        }

        scoreText.GetComponent<Text>().text = score.ToString();
    }

    public override void AgentReset()
    {
        if (invertX)
        {
            invertMult = -1f;
        }
        else
        {
            invertMult = 1f;
        }

        gameObject.transform.position = new Vector3(-invertMult * Random.Range(6f, 8f), -1.5f, 0f) + transform.parent.transform.position;
        rb.velocity = new Vector3(0f, 0f, 0f);
    }
}
