using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeAgent : Agent {
    
    public MazeAcademy academyRef;
    public int winCount = 0;
    public AutoAverage winningRate100 = new AutoAverage(100);
    private int totalGames = 0;

	public override List<float> CollectState()
	{
        List<float> state = new List<float>(academyRef.GetState());
        
		return state;
	}

	public override void AgentStep(float[] act)
	{
        int nextAction = Mathf.FloorToInt(act[0]);
        float tempRewards = academyRef.StepAction(nextAction);
        reward += tempRewards;
        if (academyRef.done)
        {
            totalGames++;
            if(academyRef.Win)
            {
                winCount++;
                winningRate100.AddValue(1);
            }
            else
            {
                winningRate100.AddValue(0);
            }
        }
	}

	public override void AgentReset()
	{

	}

	public override void AgentOnDone()
	{

	}
}
