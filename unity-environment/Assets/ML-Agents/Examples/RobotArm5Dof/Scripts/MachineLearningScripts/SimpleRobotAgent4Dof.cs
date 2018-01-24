using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRobotAgent4Dof : Agent {

    [Header("Reward Variables")]
    public int MaxScore = 1;
    public float PerStepPenalty = 0.002f;
    public float PerDegreePenalty = 0.0001f;

    [Header("Step Skip Randomization")]
    public int SkipMin = 0;
    public int SkipMax = 4;
    public int CurrentSkip = 0;

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
        if (gm.Score == MaxScore) SetDone();
    }

    public void SubtractStepPenalty()
    {
        reward -= PerStepPenalty;
    }

    public void SubtractMovementPenalty(float degrees)
    {
        reward -= PerDegreePenalty * degrees;
    }

    public override void AgentStep(float[] act)
    {
        if (CurrentSkip == 0)
        {
            CurrentSkip = Random.Range(SkipMin, SkipMax);
            if (brain.brainParameters.actionSpaceType == StateType.continuous)
            {
                SubtractStepPenalty();
                gm.ArmController.SetRotation(0, Mathf.Clamp(act[0], -1, 1));
                gm.ArmController.SetRotation(1, Mathf.Clamp(act[1], -1, 1));
                gm.ArmController.SetBend(0, Mathf.Clamp(act[2], -1, 1));
                gm.ArmController.SetBend(1, Mathf.Clamp(act[3], -1, 1));
                gm.ArmController.SetBend(2, Mathf.Clamp(act[4], -1, 1));
            }
            else
            {
                Debug.Log("This project is not set up for discrete input");
            }
        }
        else
        {
            CurrentSkip--;
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
        state.Add((gm.ArmController.Rotators[0].CurrentRotation - 180f) / 180f);

        state.Add((gm.ArmController.Rotators[1].CurrentRotation - 180f) / 180f);

        // Bend amount: -90 to 90 needs to map to -1 to 1
        var bendRange = gm.ArmController.BendMinMax.y - gm.ArmController.BendMinMax.x;
        var halfBendRange = bendRange / 2f;
        var midBend = gm.ArmController.BendMinMax.x + halfBendRange;

        var b1 = gm.ArmController.Benders[0].CurrentBend;
        state.Add((b1 - midBend) / halfBendRange);

        var b2 = gm.ArmController.Benders[1].CurrentBend;
        state.Add((b2 - midBend) / halfBendRange);

        var b3 = gm.ArmController.Benders[2].CurrentBend;
        state.Add((b3 - midBend) / halfBendRange);

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