using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAgent : Agent
{
	public GameObject goalHolder;
    public GameObject block;
    public GameObject area;
    public GameObject wall;

	public override List<float> CollectState()
	{
		List<float> state = new List<float>();
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
		state.Add(transform.position.x - area.transform.position.x);
		state.Add(transform.position.y - area.transform.position.y);
		state.Add(transform.position.z + 5 - area.transform.position.z);

        state.Add(goalHolder.transform.position.x -area.transform.position.x);
		state.Add(goalHolder.transform.position.y -area.transform.position.y);
		state.Add(goalHolder.transform.position.z + 5 -area.transform.position.z);

		state.Add(block.transform.position.x - area.transform.position.x);
        state.Add(block.transform.position.y - area.transform.position.y);
		state.Add(block.transform.position.z + 5 - area.transform.position.z);

		state.Add(wall.transform.localScale.y);

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

        Vector3 fwd = transform.TransformDirection(Vector3.down);
        if (!Physics.Raycast(transform.position, fwd, 0.55f) && 
            !Physics.Raycast(transform.position + new Vector3(0.45f, 0f, 0f), fwd, 0.55f) && 
            !Physics.Raycast(transform.position + new Vector3(-0.45f, 0f, 0f), fwd, 0.55f) && 
            !Physics.Raycast(transform.position + new Vector3(0.0f, 0f, 0.45f), fwd, 0.55f) && 
            !Physics.Raycast(transform.position + new Vector3(0.0f, 0f, -0.45f), fwd, 0.55f))
        { 
            directionY = 0f;
            directionX = directionX / 5f;
            directionZ = directionZ / 5f;
        }

		gameObject.GetComponent<Rigidbody>().AddForce(new Vector3(directionX * 40f, directionY * 300f, directionZ * 40f));
        if (GetComponent<Rigidbody>().velocity.sqrMagnitude > 25f) 
        {
            GetComponent<Rigidbody>().velocity *= 0.95f;
        }

        if (gameObject.transform.position.y < 0.0f || Mathf.Abs(gameObject.transform.position.x - area.transform.position.x) > 8f || Mathf.Abs(gameObject.transform.position.z + 5 - area.transform.position.z) > 8)
		{
			done = true;
			reward = -1f;
		}
	}

	public override void AgentReset()
	{
        transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 1.1f, -9f) + area.transform.position;
		GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        goalHolder.transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 0.1f, 0f) + area.transform.position;
        wall.transform.localScale = new Vector3(12f, Random.Range(1f, 3f), 1f);
        block.transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 1f, Random.Range(-4f, -8f)) + area.transform.position;
	}

	public override void AgentOnDone()
	{

	}
}
