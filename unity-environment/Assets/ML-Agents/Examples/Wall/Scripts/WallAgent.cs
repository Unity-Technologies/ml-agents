using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAgent : Agent
{
	public GameObject goalHolder;

	public override List<float> CollectState()
	{
		List<float> state = new List<float>();
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
		state.Add(transform.position.x);
		state.Add(transform.position.y);
		state.Add(transform.position.z);

        state.Add(velocity.x);
		state.Add(velocity.y);
		state.Add(velocity.z);

		return state;
	}

	public override void AgentStep(float[] act)
	{
        reward = -0.01f;
		float movement = act[0];
		float directionX = 0;
		float directionZ = 0;
		float directionY = 0;
		if (movement == 0) { directionX = -1; }
		if (movement == 1) { directionX = 1; }
		if (movement == 2) { directionZ = -1; }
		if (movement == 3) { directionZ = 1; }
		if (movement == 4) { directionY = 1; }
        if (gameObject.transform.position.y > 1.1f || gameObject.transform.position.y < 0.9f) 
        { 
            directionY = 0f;
            directionX = directionX / 2f;
            directionZ = directionZ / 2f;
        }
		gameObject.GetComponent<Rigidbody>().AddForce(new Vector3(directionX * 40f, directionY * 400f, directionZ * 40f));
        Debug.Log(GetComponent<Rigidbody>().velocity.sqrMagnitude);
        if (GetComponent<Rigidbody>().velocity.sqrMagnitude > 25f) 
        {
            GetComponent<Rigidbody>().velocity *= 0.95f;
        }

        if (gameObject.transform.position.y < 0.0f || Mathf.Abs(gameObject.transform.position.x) > 7f || Mathf.Abs(gameObject.transform.position.z) > 7f)
		{
			done = true;
			reward = -0.1f;
		}
	}

	public override void AgentReset()
	{
		transform.position = new Vector3(0f, 1.1f, -3f);
		GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        goalHolder.transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 0.2f, Random.Range(0f, 3.5f));
	}

	public override void AgentOnDone()
	{

	}
}
