using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
    public int timeSinceAction;

    public override void InitializeAgent()
    {
        timeSinceAction = (int)(Random.Range(0f, 3f));
    }

    public override void CollectObservations()
    {
        AddVectorObs(gameObject.transform.rotation.z);
        AddVectorObs(gameObject.transform.rotation.x);
        AddVectorObs((ball.transform.position.x - gameObject.transform.position.x));
        AddVectorObs((ball.transform.position.y - gameObject.transform.position.y));
        AddVectorObs((ball.transform.position.z - gameObject.transform.position.z));
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.x);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.y);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.z);

    }

    public override void AgentAction(float[] act)
    {
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float action_z = 2f * Mathf.Clamp(act[0], -1f, 1f);
            if ((gameObject.transform.rotation.z < 0.25f && action_z > 0f) ||
                (gameObject.transform.rotation.z > -0.25f && action_z < 0f))
            {
                gameObject.transform.Rotate(new Vector3(0, 0, 1), action_z);
            }
            float action_x = 2f * Mathf.Clamp(act[1], -1f, 1f);
            if ((gameObject.transform.rotation.x < 0.25f && action_x > 0f) ||
                (gameObject.transform.rotation.x > -0.25f && action_x < 0f))
            {
                gameObject.transform.Rotate(new Vector3(1, 0, 0), action_x);
            }
            if (!done)
            {
                reward = 0.1f;
            }
        }
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            done = true;
            reward = -1f;
        }
        Monitor.Log("value", value, MonitorType.slider, gameObject.transform);



    }

    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;
		//requestAction = true;
		//requestDecision = true;
    }

    public void FixedUpdate()
    {
        // In this example, the agent requests a decision at every step and a decision every 4 steps
        if(timeSinceAction == 3)
        {
            
            requestDecision = true;
            timeSinceAction = 0;
        }
        else{
            timeSinceAction += 1;
        }
        requestAction = true;
    }
}
