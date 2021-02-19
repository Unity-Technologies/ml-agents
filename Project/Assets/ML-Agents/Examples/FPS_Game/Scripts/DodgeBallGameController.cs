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
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
        [HideInInspector]
        public Collider Col;
    }

    private bool m_Initialized;
    public List<AgentInfo> Team0Players;
    public List<AgentInfo> Team1Players;
    public List<DodgeBall> dodgeBallsListTeamO;
    public List<DodgeBall> dodgeBallsListTeam1;
    public List<DodgeBall> AllBallsList;

    // public Dictionary<DodgeBallAgent, DodgeBallPlayer> PlayersDict = new Dictionary<DodgeBallAgent, DodgeBallPlayer>();

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
        for (int i = 0; i < NumberOfBallsToSpawn; i++)
        {
            GameObject g = Instantiate(BallPrefab, BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius,
                Quaternion.identity);
            DodgeBall db = g.GetComponent<DodgeBall>();
            AllBallsList.Add(db);
            g.SetActive(true);
        }

        //Reset Agents
        foreach (var item in Team0Players)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            // item.Col = item.Agent.GetComponent<Collider>();
        }
        foreach (var item in Team1Players)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
        }

        m_Initialized = true;
    }

    void PlayerWasHit(DodgeBallAgent agent)
    {
        // var team


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
        }
        foreach (var item in Team1Players)
        {
            item.Agent.EndEpisode();
        }

        //Reset Agents
        foreach (var item in Team0Players)
        {
            ResetAgent(item);
        }
        foreach (var item in Team1Players)
        {
            ResetAgent(item);
        }
    }

    void ResetAgent(AgentInfo item)
    {
            // var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
            // var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;
        item.Agent.transform.SetPositionAndRotation(item.StartingPos, item.StartingRot);
        item.Rb.velocity = Vector3.zero;
        item.Rb.angularVelocity = Vector3.zero;
        item.Agent.gameObject.SetActive(true);
    }

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
