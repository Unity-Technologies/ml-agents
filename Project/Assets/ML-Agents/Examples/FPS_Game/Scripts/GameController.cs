using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;
using UnityEngine.SceneManagement;
public class GameController : MonoBehaviour
{
    //NOT IMPLEMENTED YET;
    [Header("GRAPHICS")] public bool TurnOffFancyGraphics; //turns off expensive graphics/particle effects

    [Header("GLOBAL SETTINGS")]
    public List<Rigidbody> AllRBsList = new List<Rigidbody>();

    public float ExplosionForce = 100;
    public float ExplosionUpwardsModifier = 1;
    public float ExplosionRadius = 10;
    public ForceMode ExplosionForceMode;

    public bool triggerExplosion;


    [Header("SPAWN SETTINGS")]
    public GameObject BlueSpawn;
    public GameObject PurpleSpawn;
    public bool SpawnPlayer;
    public bool SpawnBaddies;

    [Header("PLAYER PREFABS")]
    public GameObject PlayerPrefab;
    public GameObject AIPrefab;
    public GameObject AITarget;
    public Transform SpawnPlatform;

    public int NumberOfEnemiesToSpawn = 3;
    public enum PlayerType
    {
        Player, AI_Heuristic, AI_Agent
    }
    public enum GameMode
    {
        SinglePlayer, PVP_Single
    }

    public GameMode gameMode;
    [Header("PLAYER DAMAGE")]
    public float DamagePerHit = 15;
    // Start is called before the first frame update
    void Awake()
    {

        if (SpawnPlayer && PlayerPrefab && BlueSpawn)
        {
            var randomPos = Random.insideUnitSphere * 3;
            randomPos.y = 0;
            var go = Instantiate(PlayerPrefab, BlueSpawn.transform.position + randomPos, quaternion.identity);
            go.SetActive(true);
            AITarget = go;
        }

        if (SpawnBaddies)
        {
            if (AIPrefab && PurpleSpawn)
            {
                var randomPos = Random.insideUnitSphere * 5;
                randomPos.y = 0;
                var go = Instantiate(AIPrefab, PurpleSpawn.transform.position + randomPos, quaternion.identity);
                go.SetActive(true);
            }
            for (int i = 0; i < NumberOfEnemiesToSpawn; i++)
            {
                if (AIPrefab && PurpleSpawn)
                {
                    var randomPos = Random.insideUnitSphere * 40;
                    randomPos.y = 3;
                    var go = Instantiate(AIPrefab, SpawnPlatform.position + randomPos, quaternion.identity);
                    go.SetActive(true);
                }
            }

        }

        Rigidbody[] rbs = Resources.FindObjectsOfTypeAll<Rigidbody>();

        foreach (var rb in rbs)
        {
            if (!rb.transform.CompareTag("projectile"))
            {
                AllRBsList.Add(rb);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (triggerExplosion)
        {
            triggerExplosion = false;
            AddExplosiveForcesToAllRB(transform.position);
        }

        //        if (Input.GetKeyDown(KeyCode.R))
        //        {
        //            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        //
        //        }

    }

    void SetupPlayer()
    {

    }

    public void AddExplosiveForcesToAllRB(Vector3 pos)
    {
        foreach (var rb in AllRBsList)
        {
            rb.AddExplosionForce(ExplosionForce, pos, ExplosionRadius, ExplosionUpwardsModifier, ExplosionForceMode);
        }
    }
}
