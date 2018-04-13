using UnityEngine;

public class ReacherAgent : Agent {

    public GameObject pendulumA;
    public GameObject pendulumB;
    public GameObject hand;
    public GameObject goal;
    float goalDegree;
    Rigidbody rbA;
    Rigidbody rbB;
    float goalSpeed;
    public float goalSize;
    ReacherAcademy myAcademy;

    /// <summary>
    /// Collect the rigidbodies of the reacher in order to resue them for 
    /// observations and actions.
    /// </summary>
    public override void InitializeAgent()
    {
        rbA = pendulumA.GetComponent<Rigidbody>();
        rbB = pendulumB.GetComponent<Rigidbody>();
        myAcademy = GameObject.Find("Academy").GetComponent<ReacherAcademy>();
    }

    /// <summary>
    /// We collect the normalized rotations, angularal velocities, and velocities of both
    /// limbs of the reacher as well as the relative position of the target and hand.
    /// </summary>
    public override void CollectObservations()
    {
        AddVectorObs(pendulumA.transform.position - transform.position);
        AddVectorObs(pendulumA.transform.rotation);
        AddVectorObs(rbA.angularVelocity);
        AddVectorObs(rbA.velocity);

        AddVectorObs(pendulumB.transform.position - transform.position);
        AddVectorObs(pendulumB.transform.rotation);
        AddVectorObs(rbB.angularVelocity);
        AddVectorObs(rbB.velocity);

        AddVectorObs(goal.transform.position - transform.position);
        AddVectorObs(hand.transform.position - transform.position);
        AddVectorObs(goalSpeed);
	}

    /// <summary>
    /// The agent's four actions correspond to torques on each of the two joints.
    /// </summary>
    public override void AgentAction(float[] vectorAction, string textAction)
	{
        goalDegree += goalSpeed;
        UpdateGoalPosition();

	    var torqueX = ScaleContinuousAction(vectorAction[0], -2f, 2f) * 100f;
	    var torqueZ = ScaleContinuousAction(vectorAction[1], -2f, 2f) * 100f;
        rbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));

	    torqueX = ScaleContinuousAction(vectorAction[2], -2f, 2f) * 100f;
	    torqueZ = ScaleContinuousAction(vectorAction[3], -2f, 2f) * 100f;
        rbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));
	}

    /// <summary>
    /// Used to move the position of the target goal around the agent.
    /// </summary>
    private void UpdateGoalPosition() {
        var radians = goalDegree * Mathf.PI / 180f;
        var goalX = 8f * Mathf.Cos(radians);
        var goalY = 8f * Mathf.Sin(radians);

        goal.transform.position = new Vector3(goalY, -1f, goalX) + transform.position;
    }

    /// <summary>
    /// Resets the position and velocity of the agent and the goal.
    /// </summary>
    public override void AgentReset()
    {
        pendulumA.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
        pendulumA.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        rbA.velocity = Vector3.zero;
        rbA.angularVelocity = Vector3.zero;

        pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
        pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        rbB.velocity = Vector3.zero;
        rbB.angularVelocity = Vector3.zero;

        goalDegree = Random.Range(0, 360);
        UpdateGoalPosition();

        goalSize = Random.Range(4f, 6f);
        goalSpeed = myAcademy.goalSpeed * Random.Range(-1f, 1f);

        goal.transform.localScale = new Vector3(goalSize, goalSize, goalSize);
    }
}
