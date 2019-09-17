using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class BasicAgent : Agent
{
    [Header("Specific to Basic")]
    private BasicAcademy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;
    int position;
    int smallGoalPosition;
    int largeGoalPosition;
    public GameObject largeGoal;
    public GameObject smallGoal;
    int minPosition;
    int maxPosition;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(BasicAcademy)) as BasicAcademy;
    }

    public override void CollectObservations()
    {
        AddVectorObs(position, 20);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        var movement = (int)vectorAction[0];
	    
		int direction = 0;
	    
		switch (movement)
		{
		    case 1:
		        direction = -1;
		        break;
		    case 2:
		        direction = 1;
		        break;
		}

	    position += direction;
        if (position < minPosition) { position = minPosition; }
        if (position > maxPosition) { position = maxPosition; }

        gameObject.transform.position = new Vector3(position - 10f, 0f, 0f);

        AddReward(-0.01f);

        if (position == smallGoalPosition)
        {
            Done();
            AddReward(0.1f);
        }

        if (position == largeGoalPosition)
        {
            Done();
            AddReward(1f);
        }
    }

    public override void AgentReset()
    {
        position = 10;
        minPosition = 0;
        maxPosition = 20;
        smallGoalPosition = 7;
        largeGoalPosition = 17;
        smallGoal.transform.position = new Vector3(smallGoalPosition - 10f, 0f, 0f);
        largeGoal.transform.position = new Vector3(largeGoalPosition - 10f, 0f, 0f);
    }

    public override void AgentOnDone()
    {

    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    private void WaitTimeInference()
    {
        if (!academy.GetIsInference())
        {
            RequestDecision();
        }
        else
        {
            if (timeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                timeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                timeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }

}
