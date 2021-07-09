// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
//
// public class DodgeBallPlatform : MonoBehaviour
// {
//
//     [Header("BALLS")] public GameObject BallPrefab;
//     public float BallSpawnRadius = 3;
//     public Transform BallSpawnPosition;
//     public int NumberOfBallsToSpawn = 10;
//
//     // Start is called before the first frame update
//     void Start()
//     {
//
//     }
//
//     void Initialize()
//     {
//
//         //SPAWN DODGE BALLS
//         for (int i = 0; i < NumberOfBallsToSpawn; i++)
//         {
//             GameObject g = Instantiate(BallPrefab, BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius,
//                 Quaternion.identity);
//             DodgeBall db = g.GetComponent<DodgeBall>();
//             AllBallsList.Add(db);
//             g.SetActive(true);
//         }
//
//         //INITIALIZE AGENTS
//         foreach (var item in Team0Players)
//         {
//             item.Agent.Initialize();
//             item.HitPointsRemaining = PlayerMaxHitPoints;
//             item.Agent.m_BehaviorParameters.TeamId = 0;
//             item.TeamID = 0;
//             PlayersDict.Add(item.Agent, item);
//         }
//         foreach (var item in Team1Players)
//         {
//             item.Agent.Initialize();
//             item.HitPointsRemaining = PlayerMaxHitPoints;
//             item.Agent.m_BehaviorParameters.TeamId = 1;
//             item.TeamID = 1;
//             PlayersDict.Add(item.Agent, item);
//         }
//
//         m_Initialized = true;
//     }
//
//     void ResetScene()
//     {
//         m_ResetTimer = 0;
//
//         //Reset Balls
//         foreach (var item in AllBallsList)
//         {
//             item.gameObject.SetActive(true);
//             item.transform.position = BallSpawnPosition.position + Random.insideUnitSphere * BallSpawnRadius;
//         }
//
//         //End Episode
//         foreach (var item in Team0Players)
//         {
//             if (item.Agent.enabled)
//             {
//                 item.Agent.EndEpisode();
//                 item.HitPointsRemaining = PlayerMaxHitPoints;
//             }
//         }
//         foreach (var item in Team1Players)
//         {
//             if (item.Agent.enabled)
//             {
//                 item.Agent.EndEpisode();
//                 item.HitPointsRemaining = PlayerMaxHitPoints;
//             }
//         }
//     // Update is called once per frame
//     void Update()
//     {
//         if (!m_Initialized)
//         {
//             Initialize();
//         }
//     }
// }
