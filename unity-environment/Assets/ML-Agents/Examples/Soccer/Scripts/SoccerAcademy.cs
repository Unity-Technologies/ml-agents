using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoccerAcademy : Academy
{

    public Brain brainStriker;
    public Brain brainGoalie;
    public Material redMaterial;
    public Material blueMaterial;
    public float spawnAreaMarginMultiplier;
    public float gravityMultiplier = 1;
    public bool randomizePlayersTeamForTraining = true;
    public int maxAgentSteps;

    public float agentRunSpeed;
    public float agentRotationSpeed;

    public float strikerPunish; //if opponents scores, the striker gets this neg reward (-1)
    public float strikerReward; //if team scores a goal they get a reward (+1)
    public float goaliePunish; //if opponents score, goalie gets this neg reward (-1)
    public float goalieReward; //if team scores, goalie gets this reward (currently 0...no reward. can play with this later)

    void Start()
    {
        Physics.gravity *= gravityMultiplier; //for soccer a multiplier of 3 looks good
    }
    public override void AcademyReset()
    {

    }

    public override void AcademyStep()
    {

    }

}
