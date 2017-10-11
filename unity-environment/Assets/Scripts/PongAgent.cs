using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongAgent : Agent {

    public int nextAction;
    public float[] currentState;
    public PongAcademy academyRef;
	public override List<float> CollectState()
	{
		List<float> state = new List<float>(currentState);
        
		return state;
	}

	public override void AgentStep(float[] act)
	{
        nextAction = Mathf.FloorToInt(act[0]);
        if (academyRef.done)
        {
            done = true;
        }

        if(academyRef.GameWinPlayer >= 0)
        {
            academyRef.done = true;
        }
	}

	public override void AgentReset()
	{

	}

	public override void AgentOnDone()
	{

	}
}
