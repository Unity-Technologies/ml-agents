using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRobotAgent4Dof : Agent {

    // Shortcuts to find the gameManager for this agent
    RobotArmGameManager4Dof _gm;
    RobotArmGameManager4Dof gm
    {
        get
        {
            if (_gm == null) _gm = GetComponentInParent<RobotArmGameManager4Dof>();
            if (_gm == null) Debug.LogError("RobotArmGameManager not populated!!");
            return _gm;
        }
    }

    public void SetDone()
    {
        done = true;
    }

    public void AddReward()
    {
        reward += gm.targetHitValue;
    }

    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            gm.ArmController.SetRotation(0, Mathf.Clamp(act[0], 0, 1));
            gm.ArmController.SetRotation(1, Mathf.Clamp(act[1], 0, 1));
            gm.ArmController.SetBend(0, Mathf.Clamp(act[2], 0, 1));
            gm.ArmController.SetBend(1, Mathf.Clamp(act[3], 0, 1));
            gm.ArmController.SetBend(2, Mathf.Clamp(act[4], 0, 1));
        }
        else
        {
            Debug.Log("This project is not set up for discrete input");
        }
    }

    public List<float> getFloatsXy(Vector3 target, float normDivisor = 1f)
    {
        var result = new List<float>();
        result.Add(target.x / normDivisor);
        result.Add(target.y / normDivisor);
        return result;
    }


    public List<float> getFloatsXyz(Vector3 target, float normDivisor = 1f)
    {
        var result = new List<float>();
        result.Add(target.x / normDivisor);
        result.Add(target.y / normDivisor);
        result.Add(target.z / normDivisor);
        return result;
    }


    public override List<float> CollectState()
    {
        // State will look something like this: [targetx, targety, targetz, rot1, bend1 ]

        var state = new List<float>();

        var targetLoc = gm.target.transform.position - gm.transform.position;
        state.AddRange(getFloatsXyz(targetLoc));

        // Rotate amount
        state.Add(gm.ArmController.Rotators[0].CurrentRotation / 360f);

        state.Add(gm.ArmController.Rotators[1].CurrentRotation / 360f);

        // Bend amount: -90 to 90
        var b1 = gm.ArmController.Benders[0].CurrentBend;
        state.Add((b1 - gm.ArmController.BendMinMax.x) / (gm.ArmController.BendMinMax.y - gm.ArmController.BendMinMax.x));

        var b2 = gm.ArmController.Benders[1].CurrentBend;
        state.Add((b2 - gm.ArmController.BendMinMax.x) / (gm.ArmController.BendMinMax.y - gm.ArmController.BendMinMax.x));

        var b3 = gm.ArmController.Benders[2].CurrentBend;
        state.Add((b3 - gm.ArmController.BendMinMax.x) / (gm.ArmController.BendMinMax.y - gm.ArmController.BendMinMax.x));

        // Get the hand location
        var handLoc = gm.ArmController.HitCenter.transform.position - gm.transform.position;
        state.AddRange(getFloatsXyz(handLoc));

        return state;
    }

    // Agent requests that the game be reset
    public override void AgentReset()
    {
        gm.ResetGame();
    }
}