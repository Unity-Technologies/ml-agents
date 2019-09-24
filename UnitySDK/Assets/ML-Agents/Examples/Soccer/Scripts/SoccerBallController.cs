using UnityEngine;

public class SoccerBallController : MonoBehaviour
{
    [HideInInspector]
    public SoccerFieldArea area;
    public AgentSoccer lastTouchedBy; //who was the last to touch the ball
    public string agentTag; //will be used to check if collided with a agent
    public string purpleGoalTag; //will be used to check if collided with red goal
    public string blueGoalTag; //will be used to check if collided with blue goal

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag(purpleGoalTag)) //ball touched red goal
        {
            area.GoalTouched(AgentSoccer.Team.Blue);
        }
        if (col.gameObject.CompareTag(blueGoalTag)) //ball touched blue goal
        {
            area.GoalTouched(AgentSoccer.Team.Purple);
        }
    }
}
