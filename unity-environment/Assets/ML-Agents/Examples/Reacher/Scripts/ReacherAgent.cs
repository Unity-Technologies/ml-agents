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

	public override List<float> CollectState()
	{
		List<float> state = new List<float>();
        state.Add(pendulumA.transform.rotation.x);
        state.Add(pendulumA.transform.rotation.y);
        state.Add(pendulumA.transform.rotation.z);
        state.Add(pendulumA.transform.rotation.w);
        state.Add(rbA.angularVelocity.x);
        state.Add(rbA.angularVelocity.y);
        state.Add(rbA.angularVelocity.z);
        state.Add(rbA.velocity.x);
        state.Add(rbA.velocity.y);
        state.Add(rbA.velocity.z);

        state.Add(pendulumB.transform.rotation.x);
        state.Add(pendulumB.transform.rotation.y);
        state.Add(pendulumB.transform.rotation.z);
        state.Add(pendulumB.transform.rotation.w);
        state.Add(rbB.angularVelocity.x);
        state.Add(rbB.angularVelocity.y);
        state.Add(rbB.angularVelocity.z);
        state.Add(rbB.velocity.x);
        state.Add(rbB.velocity.y);
        state.Add(rbB.velocity.z);

        state.Add(goal.transform.position.x - transform.position.x);
        state.Add(goal.transform.position.y - transform.position.y);
        state.Add(goal.transform.position.z - transform.position.z);

        state.Add(hand.transform.position.x - transform.position.x);
        state.Add(hand.transform.position.y - transform.position.y);
        state.Add(hand.transform.position.z - transform.position.z);


		return state;
	}

	public override void AgentStep(float[] act)
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
