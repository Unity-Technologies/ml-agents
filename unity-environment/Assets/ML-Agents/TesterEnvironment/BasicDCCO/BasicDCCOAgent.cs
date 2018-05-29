using UnityEngine;

public class BasicDCCOAgent : Agent
{
    public int position;
    public int smallGoalPosition;
    public int largeGoalPosition;
    public int minPosition;
    public int maxPosition;

    int decisionCountdown;

    public override void InitializeAgent()
    {
        decisionCountdown = Random.Range(0, 5);
    }
    public override void CollectObservations()
    {
        AddVectorObs(OneHot(position - minPosition, maxPosition - minPosition + 1));
    }

    public static float[] OneHot(int index, int size)
    {
        var res = new float[size];
        res[index] = 1f;
        return res;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var movement = (int)vectorAction[0];

        int direction = 0;

        switch (movement)
        {
            case 0:
                direction = -1;
                break;
            case 1:
                direction = 1;
                break;
        }

        position += direction;
        if (position < minPosition) { position = minPosition; }
        if (position > maxPosition) { position = maxPosition; }

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
        position = 0;
        minPosition = -10;
        maxPosition = 10;
        smallGoalPosition = -3;
        largeGoalPosition = 7;
    }

    public override void AgentOnDone()
    {

    }

	private void FixedUpdate()
	{
        if (agentParameters.onDemandDecision)
        {
            if (decisionCountdown <= 0)
            {
                decisionCountdown = Random.Range(0, 5);
                RequestDecision();
            }
            else
            {
                decisionCountdown -= 1;
            }
        }
	}
}
