using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        state.Add(gameObject.transform.rotation.z);
        state.Add(gameObject.transform.rotation.x);
        state.Add((ball.transform.position.x - gameObject.transform.position.x));
        state.Add((ball.transform.position.y - gameObject.transform.position.y));
        state.Add((ball.transform.position.z - gameObject.transform.position.z));
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.x);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.y);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.z);
        return state;
    }

    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float action_z = act[0];
            if (action_z > 2f)
            {
                action_z = 2f;
            }
            if (action_z < -2f)
            {
                action_z = -2f;
            }
            if ((gameObject.transform.rotation.z < 0.25f && action_z > 0f) ||
                (gameObject.transform.rotation.z > -0.25f && action_z < 0f))
            {
                gameObject.transform.Rotate(new Vector3(0, 0, 1), action_z);
            }
            float action_x = act[1];
            if (action_x > 2f)
            {
                action_x = 2f;
            }
            if (action_x < -2f)
            {
                action_x = -2f;
            }
            if ((gameObject.transform.rotation.x < 0.25f && action_x > 0f) ||
                (gameObject.transform.rotation.x > -0.25f && action_x < 0f))
            {
                gameObject.transform.Rotate(new Vector3(1, 0, 0), action_x);
            }
				

            if (done == false)
            {
                reward = 0.1f;
            }
        }
        else
        {
            int action = (int)act[0];
            if (action == 0 || action == 1)
            {
                action = (action * 2) - 1;
                float changeValue = action * 2f;
                if ((gameObject.transform.rotation.z < 0.25f && changeValue > 0f) ||
                    (gameObject.transform.rotation.z > -0.25f && changeValue < 0f))
                {
                    gameObject.transform.Rotate(new Vector3(0, 0, 1), changeValue);
                }
            }
            if (action == 2 || action == 3)
            {
                action = ((action - 2) * 2) - 1;
                float changeValue = action * 2f;
                if ((gameObject.transform.rotation.x < 0.25f && changeValue > 0f) ||
                    (gameObject.transform.rotation.x > -0.25f && changeValue < 0f))
                {
                    gameObject.transform.Rotate(new Vector3(1, 0, 0), changeValue);
                }
            }
            if (done == false)
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

    }

    // to be implemented by the developer
    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;
    }
}
