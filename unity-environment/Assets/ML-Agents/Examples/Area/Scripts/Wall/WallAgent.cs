using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAgent : AreaAgent
{
	public GameObject goalHolder;
    public GameObject block;
    public GameObject wall;

    public override void InitializeAgent()
    {
		base.InitializeAgent();
    }

	public override void CollectObservations()
	{
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
        AddVectorObs((transform.position.x - area.transform.position.x));
        AddVectorObs((transform.position.y - area.transform.position.y));
        AddVectorObs((transform.position.z + 5 - area.transform.position.z));

        AddVectorObs((goalHolder.transform.position.x - area.transform.position.x));
        AddVectorObs((goalHolder.transform.position.y - area.transform.position.y));
        AddVectorObs((goalHolder.transform.position.z + 5 - area.transform.position.z));

        AddVectorObs((block.transform.position.x - area.transform.position.x));
        AddVectorObs((block.transform.position.y - area.transform.position.y));
        AddVectorObs((block.transform.position.z + 5 - area.transform.position.z));

		AddVectorObs(wall.transform.localScale.y);

		AddVectorObs(velocity.x);
		AddVectorObs(velocity.y);
		AddVectorObs(velocity.z);

        Vector3 blockVelocity = block.GetComponent<Rigidbody>().velocity;
		AddVectorObs(blockVelocity.x);
		AddVectorObs(blockVelocity.y);
		AddVectorObs(blockVelocity.z);
	}

	public override void AgentAction(float[] act)
	{
        reward = -0.005f;
        MoveAgent(act);

        if (gameObject.transform.position.y < 0.0f ||
            Mathf.Abs(gameObject.transform.position.x - area.transform.position.x) > 8f ||
            Mathf.Abs(gameObject.transform.position.z + 5 - area.transform.position.z) > 8)
		{
			done = true;
			reward = -1f;
		}
	}

	public override void AgentReset()
	{
		transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 1.1f, -8f) + area.transform.position;
		GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);

        area.GetComponent<Area>().ResetArea();
	}

	public override void AgentOnDone()
	{

	}
}
