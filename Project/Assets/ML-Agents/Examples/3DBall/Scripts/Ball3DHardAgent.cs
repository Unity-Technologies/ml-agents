using UnityEngine;
using MLAgents;

public class Ball3DHardAgent : Agent
{
    [Header("Specific to Ball3DHard")]
    public GameObject ball;
    Rigidbody m_BallRb;
    IFloatProperties m_ResetParams;

    public override void InitializeAgent()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_ResetParams = Academy.Instance.FloatProperties;
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(gameObject.transform.rotation.z);
        sensor.AddObservation(gameObject.transform.rotation.x);
        sensor.AddObservation((ball.transform.position - gameObject.transform.position));
    }

    public override void AgentAction(float[] vectorAction)
    {
        var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f);

        if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
            (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
        {
            gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
        }

        if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
            (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
        {
            gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
        }
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            SetReward(-1f);
            Done();
        }
        else
        {
            SetReward(0.1f);
        }
    }

    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
            + gameObject.transform.position;
    }

    public void SetBall()
    {
        //Set the attributes of the ball by fetching the information from the academy
        m_BallRb.mass = m_ResetParams.GetPropertyWithDefault("mass", 1.0f);
        var scale = m_ResetParams.GetPropertyWithDefault("scale", 1.0f);
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetBall();
    }
}
