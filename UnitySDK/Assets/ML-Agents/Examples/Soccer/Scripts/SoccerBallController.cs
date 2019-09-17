using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoccerBallController : MonoBehaviour
{

    [HideInInspector]
    public SoccerFieldArea area;
    public AgentSoccer lastTouchedBy; //who was the last to touch the ball
    public string agentTag; //will be used to check if collided with a agent
    public string redGoalTag; //will be used to check if collided with red goal
    public string blueGoalTag; //will be used to check if collided with blue goal

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag(redGoalTag)) //ball touched red goal
        {
            area.GoalTouched(AgentSoccer.Team.Blue);
        }
        if (col.gameObject.CompareTag(blueGoalTag)) //ball touched blue goal
        {
            area.GoalTouched(AgentSoccer.Team.Red);
        }
    }
}
