using System.Collections;
using System.Collections.Generic;
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

    public override void InitializeAgent()
    {
        rbA = pendulumA.GetComponent<Rigidbody>();
        rbB = pendulumB.GetComponent<Rigidbody>();
    }

	public override void CollectObservations()
	{
        AddVectorObs(pendulumA.transform.rotation.x);
        AddVectorObs(pendulumA.transform.rotation.y);
        AddVectorObs(pendulumA.transform.rotation.z);
        AddVectorObs(pendulumA.transform.rotation.w);
        AddVectorObs(rbA.angularVelocity.x);
        AddVectorObs(rbA.angularVelocity.y);
        AddVectorObs(rbA.angularVelocity.z);
        AddVectorObs(rbA.velocity.x);
        AddVectorObs(rbA.velocity.y);
        AddVectorObs(rbA.velocity.z);

        AddVectorObs(pendulumB.transform.rotation.x);
        AddVectorObs(pendulumB.transform.rotation.y);
        AddVectorObs(pendulumB.transform.rotation.z);
        AddVectorObs(pendulumB.transform.rotation.w);
        AddVectorObs(rbB.angularVelocity.x);
        AddVectorObs(rbB.angularVelocity.y);
        AddVectorObs(rbB.angularVelocity.z);
        AddVectorObs(rbB.velocity.x);
        AddVectorObs(rbB.velocity.y);
        AddVectorObs(rbB.velocity.z);

        AddVectorObs(goal.transform.position.x - transform.position.x);
        AddVectorObs(goal.transform.position.y - transform.position.y);
        AddVectorObs(goal.transform.position.z - transform.position.z);

        AddVectorObs(hand.transform.position.x - transform.position.x);
        AddVectorObs(hand.transform.position.y - transform.position.y);
        AddVectorObs(hand.transform.position.z - transform.position.z);


	}

	public override void AgentAction(float[] act)
	{
        goalDegree += goalSpeed;
        UpdateGoalPosition();

        float torque_x = Mathf.Clamp(act[0], -1, 1) * 100f;
        float torque_z = Mathf.Clamp(act[1], -1, 1) * 100f;
        rbA.AddTorque(new Vector3(torque_x, 0f, torque_z));

        torque_x = Mathf.Clamp(act[2], -1, 1) * 100f;
        torque_z = Mathf.Clamp(act[3], -1, 1) * 100f;
        rbB.AddTorque(new Vector3(torque_x, 0f, torque_z));

	}

    void UpdateGoalPosition() {
        float radians = (goalDegree * Mathf.PI) / 180f;
        float goalX = 8f * Mathf.Cos(radians);
        float goalY = 8f * Mathf.Sin(radians);

        goal.transform.position = new Vector3(goalY, -1f, goalX) + transform.position;
    }


    public override void AgentReset()
    {
        pendulumA.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
        pendulumA.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        rbA.velocity = new Vector3(0f, 0f, 0f);
        rbA.angularVelocity = new Vector3(0f, 0f, 0f);

        pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
        pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
        rbB.velocity = new Vector3(0f, 0f, 0f);
        rbB.angularVelocity = new Vector3(0f, 0f, 0f);


        goalDegree = Random.Range(0, 360);
        UpdateGoalPosition();

        ReacherAcademy academy = GameObject.Find("Academy").GetComponent<ReacherAcademy>();
        float goalSize = academy.goalSize;
        goalSpeed = academy.goalSpeed;

        goal.transform.localScale = new Vector3(goalSize, goalSize, goalSize);
    }

	public override void AgentOnDone()
	{

	}
}
