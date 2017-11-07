using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRobotAgent : Agent {

    // Shortcuts to find the gameManager for this agent
    RobotArmGameManager _gm;
    RobotArmGameManager gm
    {
        get
        {
            if (_gm == null) _gm = GetComponentInParent<RobotArmGameManager>();
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

    enum RotationDirection { still = 0, left = 1, right = 2, up = 3, down = 4, leftup = 5, leftdown = 6, rightup = 7, rightdown = 8 };

    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {
        if (brain.brainParameters.actionSpaceType == StateType.discrete)
        {
            var action = (RotationDirection)(int)act[0];

            switch (action)
            {
                case RotationDirection.left:
                    gm.ArmController.Left();
                    break;

                case RotationDirection.right:
                    gm.ArmController.Right();
                    break;

                case RotationDirection.up:
                    gm.ArmController.Up();
                    break;

                case RotationDirection.down:
                    gm.ArmController.Down();
                    break;

                case RotationDirection.rightup:
                    gm.ArmController.Right();
                    gm.ArmController.Up();
                    break;

                case RotationDirection.rightdown:
                    gm.ArmController.Right();
                    gm.ArmController.Down();
                    break;

                case RotationDirection.leftup:
                    gm.ArmController.Left();
                    gm.ArmController.Up();
                    break;

                case RotationDirection.leftdown:
                    gm.ArmController.Left();
                    gm.ArmController.Down();
                    break;
            }
        }
        else
        {
            Debug.Log("This project is not set up for continuous input");
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

        // Bend amount
        state.Add(gm.Bender1.transform.localRotation.eulerAngles.x / 360f);

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