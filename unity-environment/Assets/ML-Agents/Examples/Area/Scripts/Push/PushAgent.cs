using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAgent : AreaAgent
{
	public GameObject goalHolder;
    public GameObject block;

    Rigidbody rb;

    Vector3 velocity;
    Vector3 blockVelocity;

    public override void InitializeAgent()
    {
		base.InitializeAgent();
        rb = GetComponent<Rigidbody>();
    }

	public override void CollectObservations()
	{
        velocity = rb.velocity;
        blockVelocity = block.GetComponent<Rigidbody>().velocity;
        AddVectorObs((transform.position.x - area.transform.position.x));
        AddVectorObs((transform.position.y - area.transform.position.y));
        AddVectorObs((transform.position.z + 5 - area.transform.position.z));

        AddVectorObs((goalHolder.transform.position.x - area.transform.position.x));
        AddVectorObs((goalHolder.transform.position.y - area.transform.position.y));
        AddVectorObs((goalHolder.transform.position.z + 5 - area.transform.position.z));

        AddVectorObs((block.transform.position.x - area.transform.position.x));
        AddVectorObs((block.transform.position.y - area.transform.position.y));
        AddVectorObs((block.transform.position.z + 5 - area.transform.position.z));

		AddVectorObs(velocity.x);
		AddVectorObs(velocity.y);
		AddVectorObs(velocity.z);

		AddVectorObs(blockVelocity.x);
		AddVectorObs(blockVelocity.y);
		AddVectorObs(blockVelocity.z);

        AddVectorObs(block.transform.localScale.x);
        AddVectorObs(goalHolder.transform.localScale.x);

	}

	public override void AgentAction(float[] act)
	{
        reward = -0.005f;
        MoveAgent(act);

        if (gameObject.transform.position.y < 0.0f || Mathf.Abs(gameObject.transform.position.x - area.transform.position.x) > 8f ||
            Mathf.Abs(gameObject.transform.position.z + 5 - area.transform.position.z) > 8)
		{
			done = true;
			reward = -1f;
		}
	}

	public override void AgentReset()
	{
        float xVariation = GameObject.Find("Academy").GetComponent<PushAcademy>().xVariation;
        transform.position = new Vector3(Random.Range(-xVariation, xVariation), 1.1f, -8f) + area.transform.position;
		rb.velocity = new Vector3(0f, 0f, 0f);

        area.GetComponent<Area>().ResetArea();
	}

	public override void AgentOnDone()
	{

	}
}
