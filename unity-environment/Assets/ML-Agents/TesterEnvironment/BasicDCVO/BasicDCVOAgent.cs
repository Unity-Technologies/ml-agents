using UnityEngine;

public class BasicDCVOAgent : Agent
{
    public Camera camera1;
    public Camera camera2;
    int position;

    public override void InitializeAgent()
    {
        
    }
    public override void CollectObservations()
    {

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var predictedPosition = (int)vectorAction[0];

        if (predictedPosition == position)
        {
            AddReward(0.1f);
        }
        else
        {
            AddReward(-0.1f);
        }

    }

    public override void AgentReset()
    {
        position = Random.Range(0, 3);
        // The two camera point randomly towards the 2 colors 
        // NOT at the position
        int multiplier = Random.Range(0, 2)*2 - 1;
        int camPos1 = (position + 3 + 1 * multiplier) % 3;
        int camPos2 = (position + 3 - 1 * multiplier) % 3;
        camera1.gameObject.transform.position = new Vector3(
            (camPos1 - 1) * 10, 0, 0);
        camera2.gameObject.transform.position = new Vector3(
            (camPos2 - 1) * 10, 0, 0);
    }

    public override void AgentOnDone()
    {

    }
}
