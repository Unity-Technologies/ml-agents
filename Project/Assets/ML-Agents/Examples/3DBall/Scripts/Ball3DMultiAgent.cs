using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class Ball3DMultiAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
    [Tooltip("Whether to use vector observation. This option should be checked " +
        "in 3DBall scene, and unchecked in Visual3DBall scene. ")]
    public bool useVecObs;
    Rigidbody m_BallRb;
    EnvironmentParameters m_ResetParams;
    [Tooltip("Specifies which reward function to use. ")]
    public Ball3DRewardType m_RewardType;

    public GameObject goal;
    [Tooltip("Specifies the radius of the goal region")]
    public float epsilon=0.25f;
    public int stepvalue=5000;
    public override void Initialize()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVecObs)
        {
            sensor.AddObservation(gameObject.transform.rotation.z);
            sensor.AddObservation(gameObject.transform.rotation.x);
            sensor.AddObservation(ball.transform.position - goal.transform.position);
            sensor.AddObservation(m_BallRb.velocity);
        }
    }

    // public void FixedUpdate()
    // {
    //     MaxStep = stepvalue;
    // }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);

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
        float reward = 0.0f;
        if (m_RewardType == Ball3DRewardType.Time)
        {
            reward = TimeReward(ball.transform.position, goal.transform.position);
        } 
        else if(m_RewardType == Ball3DRewardType.Distance)
        {
            reward = DistanceReward(ball.transform.position, goal.transform.position);
        } 
        else if(m_RewardType == Ball3DRewardType.Power)
        {
            reward = PowerReward(ball.transform.position, goal.transform.position);
        }
        SetReward(reward);
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        goal.transform.position = new Vector3(Random.Range(-2.25f, 2.25f), 3.0f, Random.Range(-2.25f, 2.25f)) + gameObject.transform.position;
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        // goal.transform.position = new Vector3(2.25f, 3.0f, 0.0f) + gameObject.transform.position;
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
            + gameObject.transform.position;
        //Reset the parameters when the Agent is reset.
        SetResetParameters();
        MaxStep = stepvalue;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    public void SetBall()
    {
        //Set the attributes of the ball by fetching the information from the academy
        m_BallRb.mass = m_ResetParams.GetWithDefault("mass", 1.0f);
        var scale = m_ResetParams.GetWithDefault("scale", 1.0f);
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetBall();
    }

    float TimeReward(Vector3 ball, Vector3 goal)
    {
        float dist = Vector3.Distance(ball, goal);
        if (dist <= epsilon)
        {
            return 1.0f;
        }
        return 0.0f;
    }

    float DistanceReward(Vector3 ball, Vector3 goal)
    {
        float dist = Vector3.Distance(ball, goal);
        return -dist;
    }

    float PowerReward(Vector3 ball, Vector3 goal)
    {
        float maxdist = 3.54f;  // assumes max distance is 2.5 - -2.5 in each dim. This is an upper bound. 
        float dist = Vector3.Distance(ball, goal);
        //distance between our actual velocity and goal velocity
        dist = Mathf.Clamp(dist, 0, maxdist);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(dist / maxdist, 2), 2);
    }

    public void setMaxStep(int value)
    {
        stepvalue = value;
        MaxStep = value;
    }

}
