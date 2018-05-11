using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MemoryDCDOAgent : Agent
{
    public int position;
    public bool goLeft;


    public override void CollectObservations()
    {
            AddVectorObs((position == 0) ? 1 : 0);
            AddVectorObs((position == 1) ? 1 : 0);
            AddVectorObs((position == 2) ? 1 : 0);
        if (position == 0)
        {
            AddVectorObs(goLeft ? 1 : 0);
        }
        else
        {
            AddVectorObs(0);
        }
        
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        int action = (int)(vectorAction[0]);
        AddReward(-0.05f);
        if (action == 0)
        {
            position = Mathf.Min(position + 1, 2);
        }
        if ((action == 1) && (position == 2))
        {
            if (goLeft)
            {
                SetReward(1f);
                Done();
            }
            else
            {
                SetReward(-1f);
                Done();
            }
        }
        if ((action == 2) && (position == 2))
        {
            if (goLeft)
            {
                SetReward(-1f);
                Done();
            }
            else
            {
                SetReward(1f);
                Done();
            }
        }
        //Debug.Log(position +"  "+ goLeft+ "  "+GetReward());
    }

    public override void AgentReset()
    {
        position = 0;
        goLeft = Random.value > 0.5f;
    }

    public override void AgentOnDone()
    {

    }
}
