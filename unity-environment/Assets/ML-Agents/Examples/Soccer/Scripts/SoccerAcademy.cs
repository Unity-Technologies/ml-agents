using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoccerAcademy : Academy {

    public Brain brainStriker;
    public Brain brainGoalie;
    public Material redMaterial;
    public Material blueMaterial;
	public float spawnAreaMarginMultiplier; //ex: .9 means 90% of spawn area will be used.... .1 margin will be left
    public float gravityMultiplier = 1; //if ball looks floaty try adjusting this. it will be multiplied by gravity.  ex: 3 means gravity (-9.81) will be multiplied by 3 so gravity will be set to -29.43. 
    public bool randomizePlayersTeamForTraining = true; //randomly pick a player's team to generalize training. i.e.: an offensive player should know how to play for either team. This can be turned off after training
    public bool randomizeFieldOrientationForTraining = true; //rotate the field after each goal to force a player to learn to find the goal wherever it is. This can be turned off after training

	public int maxAgentSteps; //max sim steps for agents. this is here so we only have to set it once and all agents can reference it

    public float agentRunSpeed; //we will be using AddForce ForceMode.VelocityChange so this should be between 1-4. Anything higher than 4 will be too fast
    public float agentRotationSpeed; //something between 10-20 should work well.


    //Rewards
    public float strikerPunish; //if opponents scores, the striker gets this neg reward (-1)
    public float strikerReward; //if team scores a goal they get a reward (+1)
    public float defenderPunish; //not currently used
    public float defenderReward; //not currently used
    public float goaliePunish; //if opponents score, goalie gets this neg reward (-1)
    public float goalieReward; //if team scores, goalie gets this reward (currently 0...no reward. can play with this later)
	ReadRewardData readRewardData;
    public float coolDownTime;

    public int currentLesson;

    void Start()
    {
        Physics.gravity *= gravityMultiplier; //for soccer a multiplier of 3 looks good
	    // readRewardData = FindObjectOfType<ReadRewardData>(); //get reward data script

    }
	public override void AcademyReset()
	{
        currentLesson = (int)resetParameters["currentLesson"];
	}

	public override void AcademyStep()
	{
        
	}

}
