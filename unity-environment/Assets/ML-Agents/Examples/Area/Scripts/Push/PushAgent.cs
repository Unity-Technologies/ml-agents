using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAgent : AreaAgent
{
	public GameObject goalHolder;
    public GameObject block;

    public override void InitializeAgent()
    {
		base.InitializeAgent();
    }

	public override List<float> CollectState()
	{
		List<float> state = new List<float>();
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
        Vector3 blockVelocity = block.GetComponent<Rigidbody>().velocity;
        state.Add((transform.position.x - area.transform.position.x));
        state.Add((transform.position.y - area.transform.position.y));
        state.Add((transform.position.z + 5 - area.transform.position.z));

        state.Add((goalHolder.transform.position.x - area.transform.position.x));
        state.Add((goalHolder.transform.position.y - area.transform.position.y));
        state.Add((goalHolder.transform.position.z + 5 - area.transform.position.z));

        state.Add((block.transform.position.x - area.transform.position.x));
        state.Add((block.transform.position.y - area.transform.position.y));
        state.Add((block.transform.position.z + 5 - area.transform.position.z));

		state.Add(velocity.x);
		state.Add(velocity.y);
		state.Add(velocity.z);

		state.Add(blockVelocity.x);
		state.Add(blockVelocity.y);
		state.Add(blockVelocity.z);

        state.Add(block.transform.localScale.x);
        state.Add(goalHolder.transform.localScale.x);

		return state;
	}

	public override void AgentStep(float[] act)
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
		GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);

        area.GetComponent<Area>().ResetArea();
	}

	public override void AgentOnDone()
	{

	}
}
