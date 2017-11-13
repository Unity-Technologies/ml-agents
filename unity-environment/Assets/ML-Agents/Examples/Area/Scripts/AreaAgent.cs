using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaAgent : Agent
{

    public GameObject area;

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        Vector3 velocity = GetComponent<Rigidbody>().velocity;

		state.Add((transform.position.x - area.transform.position.x));
		state.Add((transform.position.y - area.transform.position.y));
		state.Add((transform.position.z + 5 - area.transform.position.z));
		state.Add(velocity.x);
		state.Add(velocity.y);
		state.Add(velocity.z);
		return state;
	}

    public void MoveAgent(float[] act) {
        float directionX = 0;
        float directionZ = 0;
        float directionY = 0;

        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            directionX = Mathf.Clamp(act[0], -1f, 1f);
            directionZ = Mathf.Clamp(act[1], -1f, 1f);
            directionY = Mathf.Clamp(act[2], -1f, 1f);
            if (GetComponent<Rigidbody>().velocity.y > 0) { directionY = 0f; }
        }
        else {
            int movement = Mathf.FloorToInt(act[0]);
            if (movement == 1) { directionX = -1; }
            if (movement == 2) { directionX = 1; }
            if (movement == 3) { directionZ = -1; }
            if (movement == 4) { directionZ = 1; }
            if (movement == 5 && GetComponent<Rigidbody>().velocity.y <= 0) { directionY = 1; }
        }

        float edge = 0.499f;
        float rayDepth = 0.51f;

        Vector3 fwd = transform.TransformDirection(Vector3.down);
        if (!Physics.Raycast(transform.position, fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(edge, 0f, 0f), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(-edge, 0f, 0f), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(0.0f, 0f, edge), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(0.0f, 0f, -edge), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(edge, 0f, edge), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(-edge, 0f, edge), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(edge, 0f, -edge), fwd, rayDepth) &&
            !Physics.Raycast(transform.position + new Vector3(-edge, 0f, -edge), fwd, rayDepth))
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
		transform.position = new Vector3(Random.Range(-3.5f, 3.5f), 1.1f, -8f) + area.transform.position;
		GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);

		area.GetComponent<Area>().ResetArea();
	}

}