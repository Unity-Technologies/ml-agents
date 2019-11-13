using UnityEngine;
using MLAgents;

public class ReacherAgent : Agent
{
    public GameObject pendulumA;
    public GameObject pendulumB;
    public GameObject hand;
    public GameObject goal;
    ReacherAcademy m_MyAcademy;
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

    /// <summary>
    /// Collect the rigidbodies of the reacher in order to resue them for
    /// observations and actions.
    /// </summary>
    public override void InitializeAgent()
    {
        m_RbA = pendulumA.GetComponent<Rigidbody>();
        m_RbB = pendulumB.GetComponent<Rigidbody>();
        m_MyAcademy = GameObject.Find("Academy").GetComponent<ReacherAcademy>();

        SetResetParameters();
    }

    /// <summary>
    /// We collect the normalized rotations, angularal velocities, and velocities of both
    /// limbs of the reacher as well as the relative position of the target and hand.
    /// </summary>
    public override void CollectObservations()
    {
        AddVectorObs(pendulumA.transform.localPosition);
        AddVectorObs(pendulumA.transform.rotation);
        AddVectorObs(m_RbA.angularVelocity);
        AddVectorObs(m_RbA.velocity);

        AddVectorObs(pendulumB.transform.localPosition);
        AddVectorObs(pendulumB.transform.rotation);
        AddVectorObs(m_RbB.angularVelocity);
        AddVectorObs(m_RbB.velocity);

        AddVectorObs(goal.transform.localPosition);
        AddVectorObs(hand.transform.localPosition);

        AddVectorObs(m_GoalSpeed);
    }

    /// <summary>
    /// The agent's four actions correspond to torques on each of the two joints.
    /// </summary>
    public override void AgentAction(float[] vectorAction)
    {
        m_GoalDegree += m_GoalSpeed;
        UpdateGoalPosition();

        var torqueX = Mathf.Clamp(vectorAction[0], -1f, 1f) * 150f;
        var torqueZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * 150f;
        m_RbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));

        torqueX = Mathf.Clamp(vectorAction[2], -1f, 1f) * 150f;
        torqueZ = Mathf.Clamp(vectorAction[3], -1f, 1f) * 150f;
        m_RbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));
    }

    /// <summary>
    /// Used to move the position of the target goal around the agent.
    /// </summary>
    void UpdateGoalPosition()
    {
        var radians = m_GoalDegree * Mathf.PI / 180f;
        var goalX = 8f * Mathf.Cos(radians);
        var goalY = 8f * Mathf.Sin(radians);
        var goalZ = m_Deviation * Mathf.Cos(m_DeviationFreq * radians);
        goal.transform.position = new Vector3(goalY, goalZ, goalX) + transform.position;
    }

    /// <summary>
    /// Resets the position and velocity of the agent and the goal.
    /// </summary>
    public override void AgentReset()
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
        UpdateGoalPosition();

        SetResetParameters();


        goal.transform.localScale = new Vector3(m_GoalSize, m_GoalSize, m_GoalSize);
    }

    public void SetResetParameters()
    {
        m_GoalSize = m_MyAcademy.resetParameters["goal_size"];
        m_GoalSpeed = Random.Range(-1f, 1f) * m_MyAcademy.resetParameters["goal_speed"];
        m_Deviation = m_MyAcademy.resetParameters["deviation"];
        m_DeviationFreq = m_MyAcademy.resetParameters["deviation_freq"];
    }
}
