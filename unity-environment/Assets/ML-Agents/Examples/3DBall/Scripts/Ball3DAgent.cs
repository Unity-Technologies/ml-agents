using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;

    public override void InitializeAgent()
    {

    }

    public override void CollectObservations()
    {
        AddVectorObs(gameObject.transform.rotation.z);
        AddVectorObs(gameObject.transform.rotation.x);
        AddVectorObs(ball.transform.position - gameObject.transform.position);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            var actionZ = ScaleContinuousAction(vectorAction[0], -2f, 2f);
            if (gameObject.transform.rotation.z < 0.25f && actionZ > 0f ||
                gameObject.transform.rotation.z > -0.25f && actionZ < 0f)
            {
                gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
            }

            var actionX = ScaleContinuousAction(vectorAction[1], -2f, 2f);
            if (gameObject.transform.rotation.x < 0.25f && actionX > 0f ||
                gameObject.transform.rotation.x > -0.25f && actionX < 0f)
            {
                gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
            }

            SetReward(0.1f);

        }
        if (ball.transform.position.y - gameObject.transform.position.y < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            Done();
            SetReward(-1f);
        }


    }

    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        ball.GetComponent<Rigidbody>().velocity = Vector3.zero;
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;

    }

}
