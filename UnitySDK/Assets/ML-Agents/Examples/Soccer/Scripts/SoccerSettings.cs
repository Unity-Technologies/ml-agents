using UnityEngine;

public class SoccerSettings : MonoBehaviour
{
    public Material purpleMaterial;
    public Material blueMaterial;
    public bool randomizePlayersTeamForTraining = true;

    public float agentRunSpeed;

    public float strikerPunish; //if opponents scores, the striker gets this neg reward (-1)
    public float strikerReward; //if team scores a goal they get a reward (+1)
    public float goaliePunish; //if opponents score, goalie gets this neg reward (-1)
    public float goalieReward; //if team scores, goalie gets this reward (currently 0...no reward. can play with this later)
}
