using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BouncerAgent : Agent {

    [Header("Bouncer Specific")]
    public GameObject banana;
    Rigidbody rb;
    float speed = 10f;

    public override void InitializeAgent()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        rb.velocity = new Vector3((1 - 2 * Random.value) * 15, 0, (1 - 2 * Random.value) * 15);
        rb.velocity = rb.velocity.normalized * speed;
    }

	public override void CollectObservations()
	{
        AddVectorObs(gameObject.transform.position.x / 25f);
        AddVectorObs(gameObject.transform.position.z / 25f);
        AddVectorObs(banana.transform.position.x / 25f);
        AddVectorObs(banana.transform.position.z / 25f);
	}

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        float x = Mathf.Clamp(vectorAction[0], -1, 1);
        float z = Mathf.Clamp(vectorAction[1], -1, 1);
        rb.velocity = new Vector3(x, 0, z) ;
        if (rb.velocity.magnitude < 0.01f){
            AddReward(-1);
            Done();
            return;
        }
        rb.velocity = rb.velocity.normalized * speed;
        if ((gameObject.transform.position.x + rb.velocity.x * 2 > 24)
            || (gameObject.transform.position.z + rb.velocity.z * 2 > 24)
            || (gameObject.transform.position.x + rb.velocity.x * 2 < -24)
            || (gameObject.transform.position.z + rb.velocity.z * 2 < -24))
        {
            Done();
            AddReward(-1);
        }
        else
        {
            //AddReward(0.05f);
        }
	}

	public override void AgentReset()
	{

        Vector3 oldPosition = gameObject.transform.position;
        gameObject.transform.position = new Vector3((1 - 2 * Random.value) * 15, oldPosition.y, (1 - 2 * Random.value) * 15);
        rb.velocity = new Vector3((1 - 2 * Random.value) * 15, 0, (1 - 2 * Random.value) * 15);
        rb.velocity = rb.velocity.normalized * speed;
	}

	public override void AgentOnDone()
	{

	}

    private void OnTriggerEnter(Collider collision)
    {
        if ( collision.gameObject.name.Contains("Wall")){
            RequestDecision();
        }
    }
}
