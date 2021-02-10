using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class DodgeBallGameController : MonoBehaviour
{

    [Header("PLAYERS")]
    public Transform Team0SpawnPos;
    public Transform Team1SpawnPos;
    public int NumberOfPlayersOnTeam0 = 1;
    public int NumberOfPlayersOnTeam1 = 1;

    [Header("BALLS")] public GameObject BallPrefab;
    public float BallSpawnRadius = 3;
    public Transform BallSpawnPosition;
    public int NumberOfBallsToSpawn = 10;
    public int NumberOfBallsPlayersCanHold = 3;

    [Serializable]
    public class DodgeBallPlayer
    {
        public int teamID;
        public int currentNumberOfBalls;
        public FPSAgent agent;
        public List<DodgeBall> currentlyHeldBalls;

    }

    void ResetScene()
    {
        for (int i = 0; i < NumberOfBallsToSpawn; i++)
        {
            GameObject g = Instantiate(BallPrefab, BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius,
                Quaternion.identity);
            g.SetActive(true);
        }
    }


    public List<DodgeBallAgent> Team0Players;
    public List<DodgeBallAgent> Team1Players;
    public List<DodgeBall> dodgeBallsListTeamO;
    public List<DodgeBall> dodgeBallsListTeam1;

    public Dictionary<DodgeBallAgent, DodgeBallPlayer> PlayersDict = new Dictionary<DodgeBallAgent, DodgeBallPlayer>();


    // Start is called before the first frame update
    void Awake()
    {
        ResetScene();
    }

    // // Update is called once per frame
    // void Update()
    // {
    //
    // }
}
