using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRobotAgentContinuous : Agent {

    // Shortcuts to find the gameManager for this agent
    RobotArmGameManagerContinuous _gm;
    RobotArmGameManagerContinuous gm
    {
        get
        {
            if (_gm == null) _gm = GetComponentInParent<RobotArmGameManagerContinuous>();
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
            var actionRot = Mathf.Clamp(act[0], 0, 1);
            var actionBend = Mathf.Clamp(act[1], 0, 1);

            gm.ArmController.SetRotation(actionRot);
            gm.ArmController.SetBend(actionBend);
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
        state.Add(gm.Rotator1.transform.localRotation.eulerAngles.y / 360f);

        // Bend amount: -90 to 90
        var xDeg = gm.Bender1.transform.localRotation.eulerAngles.x;
        if (xDeg > 180f) xDeg -= 360;
        state.Add(xDeg / 360f);

        var handLoc = gm.Hand.transform.position - gm.transform.position;
        state.AddRange(getFloatsXyz(handLoc));

        return state;
    }

    // Agent requests that the game be reset
    public override void AgentReset()
    {
        gm.ResetGame();
    }
}