using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class NewReacherAgent : Agent
{
    public GameObject pendulumA;
    public GameObject pendulumB;
    public GameObject hand;
    public GameObject goal;
    public GameObject ballType;
    public float ballRange;
    public int ballNumber;
    float m_GoalDegree;
    Rigidbody m_RbA;
    Rigidbody m_RbB;
    // speed of the goal zone around the arm (in radians)
    float m_GoalSpeed;
    // radius of the goal zone
    float m_GoalSize;
    // Magnitude of sinusoidal (cosine) deviation of the goal along the vertical dimension
    float m_Deviation;
    // Frequency of the cosine deviation of the goal along the vertical dimension
    float m_DeviationFreq;
    List<GameObject> m_balls;
    List<float> m_BallDegrees;
    float m_BallSpeed;
    EnvironmentParameters m_ResetParams;

    /// <summary>
    /// Collect the rigidbodies of the reacher in order to resue them for
    /// observations and actions.
    /// </summary>
    public override void Initialize()
    {
        m_RbA = pendulumA.GetComponent<Rigidbody>();
        m_RbB = pendulumB.GetComponent<Rigidbody>();

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();

        m_balls = new List<GameObject>();
        m_BallDegrees = new List<float>();
        CreateBalls();
        UpdateBallsPosition();
    }

    /// <summary>
    /// We collect the normalized rotations, angularal velocities, and velocities of both
    /// limbs of the reacher as well as the relative position of the target and hand.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(pendulumB.transform.localPosition);
        sensor.AddObservation(pendulumB.transform.rotation);
        sensor.AddObservation(m_RbB.angularVelocity);
        sensor.AddObservation(m_RbB.velocity);

        sensor.AddObservation(goal.transform.localPosition);
        sensor.AddObservation(hand.transform.localPosition);

        sensor.AddObservation(pendulumA.transform.localPosition);
        sensor.AddObservation(pendulumA.transform.rotation);
        sensor.AddObservation(m_RbA.angularVelocity);
        sensor.AddObservation(m_RbA.velocity);

        sensor.AddObservation(m_GoalSpeed);

        // irrelevant observations
        for ( int i = 0; i < ballNumber; i++)
        {
            sensor.AddObservation(m_balls[i].transform.localPosition);
        } 
        sensor.AddObservation(m_BallSpeed);
    }

    /// <summary>
    /// The agent's four actions correspond to torques on each of the two joints.
    /// </summary>
    public override void OnActionReceived(float[] vectorAction)
    {
        m_GoalDegree += m_GoalSpeed;
        for ( int i = 0; i < ballNumber; i++)
        {
            m_BallDegrees[i] += m_BallSpeed;
        }

        UpdateGoalPosition();
        UpdateBallsPosition();

        var torqueX = Mathf.Clamp(vectorAction[0], -1f, 1f) * 150f;
        var torqueZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * 150f;
        m_RbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));
        
        torqueX = Mathf.Clamp(vectorAction[2], -1f, 1f) * 150f;
        torqueZ = Mathf.Clamp(vectorAction[3], -1f, 1f) * 150f;
        m_RbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));

        // AddReward( - 0.001f * (vectorAction[0] * vectorAction[0] 
        //         + vectorAction[1] * vectorAction[1] 
        //         + vectorAction[2] * vectorAction[2] 
        //         + vectorAction[3] * vectorAction[3] 
        // ));
    }

    /// <summary>
    /// Used to move the position of the target goal around the agent.
    /// </summary>
    void UpdateGoalPosition()
    {
        // if ((goal.transform.position - hand.transform.position).magnitude > 3.5f)
        // {
        //     AddReward(-0.002f);
        // }
        AddReward( - 0.001f * (goal.transform.position - hand.transform.position).magnitude);
        // Debug.Log((goal.transform.position - hand.transform.position).magnitude);
        var radians = m_GoalDegree * Mathf.PI / 180f;
        var goalX = 8f * Mathf.Cos(radians);
        var goalY = 8f * Mathf.Sin(radians);
        var goalZ = m_Deviation * Mathf.Cos(m_DeviationFreq * radians);
        goal.transform.position = new Vector3(goalY, goalZ, goalX) + transform.position;
    }

    void UpdateBallsPosition()
    {
        for (int i = 0; i < ballNumber; i++)
        {
            var radians = m_BallDegrees[i] * Mathf.PI / 180f;
            var ballX = 8f * Mathf.Cos(radians);
            var ballY = 8f * Mathf.Sin(radians);
            // var ballZ = m_Deviation * Mathf.Cos(m_DeviationFreq * radians);
            var ballZ = 10f;
            m_balls[i].transform.position = new Vector3(ballY, ballZ, ballX) + transform.position;
        }
        
    }

    void CreateBalls()
    {
        for (int i = 0; i < ballNumber; i++)
        {
            GameObject b = Instantiate(ballType);
            m_balls.Add(b);
        }
        for (int i=0; i < ballNumber; i++)
        {
            m_BallDegrees.Add(Random.Range(0, 360));
        }
    }

    /// <summary>
    /// Resets the position and velocity of the agent and the goal.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        pendulumA.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
        pendulumA.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        m_RbA.velocity = Vector3.zero;
        m_RbA.angularVelocity = Vector3.zero;

        pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
        pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        m_RbB.velocity = Vector3.zero;
        m_RbB.angularVelocity = Vector3.zero;

        m_GoalDegree = Random.Range(0, 360);
        for (int i=0; i < ballNumber; i++)
        {
            m_BallDegrees[i] = Random.Range(0, 360);
        }

        UpdateGoalPosition();
        UpdateBallsPosition();

        SetResetParameters();


        goal.transform.localScale = new Vector3(m_GoalSize, m_GoalSize, m_GoalSize);
    }

    public void SetResetParameters()
    {
        m_GoalSize = m_ResetParams.GetWithDefault("goal_size", 5);
        m_GoalSpeed = Random.Range(-1f, 1f) * m_ResetParams.GetWithDefault("goal_speed", 1);
        m_Deviation = m_ResetParams.GetWithDefault("deviation", 0);
        m_DeviationFreq = m_ResetParams.GetWithDefault("deviation_freq", 0);
        m_BallSpeed = Random.Range(-0.5f, 0.5f) * m_ResetParams.GetWithDefault("ball_speed", 1);
    }
}
