using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class DodgeBallGameController : MonoBehaviour
{

    // public enum CurrentGameMode
    // {
    //     OneVsOne, "2v2", "3v3"
    // }
    //
    // public CurrentGameMode GameMode = CurrentGameMode.1v1;

    [Header("PLAYERS")]
    public Transform Team0SpawnPos;
    public Transform Team1SpawnPos;
    public int NumberOfPlayersOnTeam0 = 1;
    public int NumberOfPlayersOnTeam1 = 1;

    [Header("BALLS")] public GameObject BallPrefab;
    public float BallSpawnRadius = 3;
    public Transform BallSpawnPosition;
    public int NumberOfBallsToSpawn = 10;
    // public int NumberOfBallsPlayersCanHold = 4;
    public int PlayerMaxHitPoints = 5;

    // [Serializable]
    // public class DodgeBallPlayer
    // {
    //     public int teamID;
    //     public int currentNumberOfBalls;
    //     public FPSAgent agent;
    //     public List<DodgeBall> currentlyHeldBalls;
    // }

    [Serializable]
    public class AgentInfo
    {
        public DodgeBallAgent Agent;
        public int HitPointsRemaining;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
        [HideInInspector]
        public Collider Col;
        [HideInInspector]
        public int TeamID;
    }

    private bool m_Initialized;
    public List<AgentInfo> Team0Players;
    public Color Team0Color;
    public List<AgentInfo> Team1Players;
    public Color Team1Color;
    // public List<DodgeBall> dodgeBallsListTeamO;
    // public List<DodgeBall> dodgeBallsListTeam1;
    public List<DodgeBall> AllBallsList;

    public Dictionary<DodgeBallAgent, AgentInfo> PlayersDict = new Dictionary<DodgeBallAgent, AgentInfo>();

    private int m_ResetTimer;
    public int MaxEnvironmentSteps = 5000;
    void FixedUpdate()
    {
        if (m_ResetTimer > MaxEnvironmentSteps)
        {
            ResetScene();
        }
        m_ResetTimer += 1;
    }
    void Initialize()
    {

        //SPAWN DODGE BALLS
        for (int i = 0; i < NumberOfBallsToSpawn; i++)
        {
            GameObject g = Instantiate(BallPrefab, BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius,
                Quaternion.identity);
            DodgeBall db = g.GetComponent<DodgeBall>();
            AllBallsList.Add(db);
            g.SetActive(true);
        }

        //INITIALIZE AGENTS
        foreach (var item in Team0Players)
        {
            item.Agent.Initialize();
            item.HitPointsRemaining = PlayerMaxHitPoints;
            item.Agent.m_BehaviorParameters.TeamId = 0;
            item.TeamID = 0;
            PlayersDict.Add(item.Agent, item);
        }
        foreach (var item in Team1Players)
        {
            item.Agent.Initialize();
            item.HitPointsRemaining = PlayerMaxHitPoints;
            item.Agent.m_BehaviorParameters.TeamId = 1;
            item.TeamID = 1;
            PlayersDict.Add(item.Agent, item);
        }

        m_Initialized = true;
    }

    public void PlayerWasHit(DodgeBallAgent agent)
    {
        //SET AGENT/TEAM REWARDS HERE
        AgentInfo info = PlayersDict[agent];
        int hitTeamID = info.TeamID;
        // var HitTeamList = hitTeamID == 0 ? Team0Players : Team1Players;
        var HitByTeamList = hitTeamID == 1 ? Team0Players : Team1Players;
        // int hitByTeamID = hitTeamID == 0? 1: 0; //assumes only 2 teams

        if (info.HitPointsRemaining == 1)
        {
            //RESET ENV
            print($"{agent.name} Lost.{agent.name} was weak:");
            //ASSIGN REWARDS
            // EndEpisode();
            agent.AddReward(-1f); //you lost penalty
            HitByTeamList[0].Agent.AddReward(1);
            if (info.TeamID == 0)
            {
                print($"Team 1 Won");
            }
            else if (info.TeamID == 1)
            {
                print($"Team 0 Won");
            }
            ResetScene();
        }
        else
        {
            info.HitPointsRemaining--;
            //ASSIGN REWARDS
            agent.AddReward(-.1f); //small hit penalty
            HitByTeamList[0].Agent.AddReward(.1f);
        }

    }



    void ResetScene()
    {
        m_ResetTimer = 0;

        //Reset Balls
        foreach (var item in AllBallsList)
        {
            item.gameObject.SetActive(true);
            item.transform.position = BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius;
        }

        //End Episode
        foreach (var item in Team0Players)
        {
            item.Agent.EndEpisode();
            item.HitPointsRemaining = PlayerMaxHitPoints;
        }
        foreach (var item in Team1Players)
        {
            item.Agent.EndEpisode();
            item.HitPointsRemaining = PlayerMaxHitPoints;
        }

        // //Reset Agents
        // foreach (var item in Team0Players)
        // {
        //     item.Agent.ResetAgent();
        //     item.HitPointsRemaining = PlayerMaxHitPoints;
        //     // ResetAgent(item);
        // }
        // foreach (var item in Team1Players)
        // {
        //     item.Agent.ResetAgent();
        //     item.HitPointsRemaining = PlayerMaxHitPoints;
        //     // ResetAgent(item);
        // }
    }

    // void ResetAgent(AgentInfo item)
    // {
    //         // var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
    //         // var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;
    //     item.Agent.transform.SetPositionAndRotation(item.StartingPos, item.StartingRot);
    //     item.Rb.velocity = Vector3.zero;
    //     item.Rb.angularVelocity = Vector3.zero;
    //     item.Agent.gameObject.SetActive(true);
    // }

    // // Start is called before the first frame update
    // void Awake()
    // {
    //     ResetScene();
    // }

    // Update is called once per frame
    void Update()
    {
        if (!m_Initialized)
        {
            Initialize();
        }
    }
}
