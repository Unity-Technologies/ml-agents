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
    private Text textComponent;
    public GameObject myArea;
    public GameObject opponent;
    private Rigidbody agentRb;
    private Rigidbody ballRb;

    public override void InitializeAgent()
    {
        agentRb = GetComponent<Rigidbody>();
        ballRb = GetComponent<Rigidbody>();
        textComponent = scoreText.GetComponent<Text>();
    }

    public override void CollectObservations()
    {
        AddVectorObs(invertMult * (transform.position.x - myArea.transform.position.x));
        AddVectorObs(transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * agentRb.velocity.x);
        AddVectorObs(agentRb.velocity.y);

        AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * ballRb.velocity.x);
        AddVectorObs(ballRb.velocity.y);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = 0.2f * Mathf.Clamp(vectorAction[0], -3f, 3f) * invertMult;
        var moveY = Mathf.Clamp(vectorAction[1], -3f, 3f);
        
        if (moveY > 0 && transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            agentRb.velocity = new Vector3(agentRb.velocity.x, 6f, 0f);
        }

        agentRb.velocity = new Vector3(moveX * 50f, agentRb.velocity.y, 0f);

        if ((invertX && (transform.position.x - transform.parent.transform.position.x < -invertMult)) || 
            (!invertX && (transform.position.x - transform.parent.transform.position.x > -invertMult)))
        {
                transform.position = new Vector3(-invertMult + transform.parent.transform.position.x, 
                                                            transform.position.y, 
                                                            transform.position.z);
        }

        textComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        invertMult = invertX ? -1f : 1f;

        transform.position = new Vector3(-invertMult * Random.Range(6f, 8f), -1.5f, 0f) + transform.parent.transform.position;
        agentRb.velocity = new Vector3(0f, 0f, 0f);
    }
}
