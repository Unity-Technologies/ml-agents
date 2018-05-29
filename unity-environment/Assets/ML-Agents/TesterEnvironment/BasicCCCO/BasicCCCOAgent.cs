using UnityEngine;

public class BasicCCCOAgent : Agent
{
    public Vector2 position;
    public Vector2 goalPosition;
    public float maxDistance = 10f;

    int decisionCountdown;

    public override void InitializeAgent()
    {
        decisionCountdown = Random.Range(0, 5);
    }
    public override void CollectObservations()
    {
        AddVectorObs(goalPosition-position);
        AddVectorObs((position));
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        position = position + new Vector2(vectorAction[0], vectorAction[1]);
        if ((position- goalPosition).magnitude<1f)
        {
            Done();
            AddReward(1f);
        }
        else if (position.magnitude > maxDistance)
        {
            Done();
            AddReward(-1f);
        }
        else
        {
            AddReward(-0.01f);
        }
    }

    public override void AgentReset()
    {
        position = default(Vector2);
        goalPosition = new Vector2((Random.value * 2 - 1) * 5f, (Random.value * 2 - 1) * 5f);
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
