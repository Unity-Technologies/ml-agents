using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class PlayerAgent : Agent
{
	public Rigidbody rBall;
	public Transform Target;
	public Transform Aimpoint;
	public float MoveSpeed = 0.1f;
	public float ForceMultiplier = 10;
	
	public bool useVectorObs;

	Vector3 ballInitPos;
	Vector3 aimInitPos;
	bool holdBall;
	bool win;

    void Start()
    {
        rBall.useGravity = false;

        ballInitPos = rBall.transform.localPosition;
        aimInitPos = Aimpoint.localPosition;

        holdBall = true;
        win = false;
    }

    public override void OnEpisodeBegin()
    {
    	if (rBall.transform.localPosition.y <= 0.6f)
        {
            rBall.angularVelocity = Vector3.zero;
            rBall.velocity = Vector3.zero;
            rBall.transform.localPosition = ballInitPos;
            rBall.useGravity = false;
        }

        Aimpoint.localPosition = aimInitPos;

        holdBall = true;
        win = false;

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 6 - 3,
                                           Random.value * 4 - 2,
                                           Random.value * 2 + 2);
    }

	public override void CollectObservations(VectorSensor sensor)
	{
		if (useVectorObs)
		{
			sensor.AddObservation(Target.localPosition);
			sensor.AddObservation(Aimpoint.localPosition.x);
			sensor.AddObservation(Aimpoint.localPosition.y);
			sensor.AddObservation(rBall.transform.localPosition);
			sensor.AddObservation(rBall.velocity);
		}
	}

	public override void OnActionReceived(float[] vectorAction)
	{
		AddReward(-0.001f);

		// Move aimpoint
	    Vector3 aimMove = Vector3.zero;
	    aimMove.x = vectorAction[0]*MoveSpeed;
	    aimMove.y = vectorAction[1]*MoveSpeed;
	    Aimpoint.localPosition += aimMove;
	    // Vector3 x = (Aimpoint.localPosition - this.transform.localPosition);
    	// Debug.Log("x");
    	// Debug.Log(x);

	    // Shoot ball
	    bool shootBall = vectorAction[2] >= 0.5;
	    if (shootBall && holdBall) 
	    {
	    	rBall.useGravity = true;
	    	// Vector3 shootSignal = (Aimpoint.localPosition - this.transform.localPosition);
	    	Vector3 origin = Vector3.zero;
	    	origin.z = -10;
	    	Vector3 shootSignal = (Aimpoint.localPosition - origin);

	    	rBall.AddForce(shootSignal * ForceMultiplier);
	    	holdBall = false;
	    }
	    
	    // Reached target
	    // if (ScoreArea.GetComponent<ScoreArea>().Hit)
	    // {
	    //     SetReward(1.0f);
	    // 	Debug.Log("Hit!");
	    // 	win = true;
	    // 	ScoreArea.GetComponent<ScoreArea>().Hit = false;
	    // }

	    // Fell off platform
	    if (rBall.transform.localPosition.y <= 0.5f)
	    {
	    	if (!win)
	    	{
	    		// SetReward(-1.0f);
				AddReward(-1.0f);
	    	}
	        EndEpisode();
	    }
	}

	public override void Heuristic(float[] actionsOut)
	{
	    actionsOut[0] = Input.GetAxis("Horizontal");
	    actionsOut[1] = Input.GetAxis("Vertical");
	    actionsOut[2] = Input.GetAxis("Jump");
	}

	public void Hit()
	{
	    // SetReward(1.0f);
		AddReward(1.0f);
	    win = true;
	    // Debug.Log("Hit!");
	}
}
