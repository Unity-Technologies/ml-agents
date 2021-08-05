using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class DungeonEscapeEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public PushAgentEscape Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
        [HideInInspector]
        public Collider Col;
    }

    [System.Serializable]
    public class DragonInfo
    {
        public SimpleNPC Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
        [HideInInspector]
        public Collider Col;
        public Transform T;
        public bool IsDead;
    }

    EnvironmentParameters m_ResetParams;
    float m_Absorbing;

    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;
    private int m_ResetTimer;

    /// <summary>
    /// The area bounds.
    /// </summary>
    [HideInInspector]
    public Bounds areaBounds;
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;

    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();
    public List<DragonInfo> DragonsList = new List<DragonInfo>();
    private Dictionary<PushAgentEscape, PlayerInfo> m_PlayerDict = new Dictionary<PushAgentEscape, PlayerInfo>();
    public bool UseRandomAgentRotation = true;
    public bool UseRandomAgentPosition = true;
    PushBlockSettings m_PushBlockSettings;

    private int m_NumberOfRemainingPlayers;
    public GameObject Key;
    //public GameObject Tombstone;
    private SimpleMultiAgentGroup m_AgentGroup;
    private StatsRecorder m_StatsRecorder;
    void Start()
    {
        m_StatsRecorder = Academy.Instance.StatsRecorder;

        // Get the ground's bounds
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        areaBounds = ground.GetComponent<Collider>().bounds;
        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();

        //Reset Players Remaining
        m_NumberOfRemainingPlayers = AgentsList.Count;

        //Hide The Key
        Key.SetActive(false);

        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            item.Col = item.Agent.GetComponent<Collider>();
            // Add to team manager
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        foreach (var item in DragonsList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.T = item.Agent.transform;
            item.Col = item.Agent.GetComponent<Collider>();
        }

        ResetScene();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
    }

    private void SendEndStats(bool success, float reward)
    {
        m_StatsRecorder.Add("Environment/Actual Group Reward", reward);
        var successVal = success ? 1.0f : 0.0f;
        m_StatsRecorder.Add("Environment/Success rate", successVal);
    }

    public void TouchedHazard(PushAgentEscape agent)
    {
        m_NumberOfRemainingPlayers--;
        if (m_NumberOfRemainingPlayers == 0 || agent.IHaveAKey)
        {
            m_AgentGroup.EndGroupEpisode();
            SendEndStats(false, 0.0f);
            ResetScene();
        }
        else
        {
            if (m_Absorbing == 0.0f)
            {
                agent.gameObject.SetActive(false);
            }
            else
            {
                Vector3 pos = new Vector3(Random.Range(200f, 2000f), Random.Range(-1000f, 1000f), Random.Range(-1000f, 1000f));
                var rot = Quaternion.Euler(Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f));
                agent.transform.SetPositionAndRotation(pos, rot);
            }
        }
    }

    public void UnlockDoor()
    {
        m_AgentGroup.AddGroupReward(1f);
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));

        print("Unlocked Door");
        m_AgentGroup.EndGroupEpisode();
        SendEndStats(true, 1.0f);

        ResetScene();
    }

    public Vector3 GetNearestAgent(Vector3 baddiePosition)
    {
        Vector3 nearest = Vector3.zero;
        float min_dist = 50000;
        foreach (var item in AgentsList)
        {
            if (item.Agent.gameObject.activeSelf)
            {
                var dist = Vector3.Distance(baddiePosition, item.Agent.transform.position);
                if (dist < min_dist)
                {
                    nearest = item.Agent.transform.position;
                    min_dist = dist;
                }
            }
        }
        return nearest;
    }

    public void KilledByBaddie(PushAgentEscape agent, Collision baddieCol)
    {
        var baddie = baddieCol.gameObject.GetComponent<SimpleNPC>();
        if (baddie.KeyCarrier)
        {
            baddieCol.gameObject.SetActive(false);
            Key.transform.SetPositionAndRotation(baddieCol.collider.transform.position, baddieCol.collider.transform.rotation);
            Key.SetActive(true);
        }

        m_NumberOfRemainingPlayers--;
        if (m_NumberOfRemainingPlayers == 0 || agent.IHaveAKey)
        {
            m_AgentGroup.EndGroupEpisode();
            SendEndStats(false, 0.0f);
            ResetScene();
        }
        else
        {
            if (m_Absorbing == 0.0f)
            {
                agent.gameObject.SetActive(false);
            }
            else
            {
                Vector3 pos = new Vector3(Random.Range(200f, 2000f), Random.Range(-1000f, 1000f), Random.Range(-1000f, 1000f));
                var rot = Quaternion.Euler(Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f));
                agent.transform.SetPositionAndRotation(pos, rot);
            }
        }

        print($"{baddieCol.gameObject.name} ate {agent.transform.name}");

    }

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(2.5f, 0.01f, 2.5f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
    }

    /// <summary>
    /// Swap ground material, wait time seconds, then swap back to the regular material.
    /// </summary>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void BaddieTouchedBlock()
    {
        m_AgentGroup.EndGroupEpisode();

        // Swap ground material for a bit to indicate we scored.
        //StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.failMaterial, 0.5f));
        SendEndStats(false, 0.0f);
        ResetScene();
    }

    Quaternion GetRandomRot()
    {
        return Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
    }

    void ResetScene()
    {

        m_Absorbing = m_ResetParams.GetWithDefault("absorbing_state", 0.0f);
        //Reset counter
        m_ResetTimer = 0;

        //Reset Players Remaining
        m_NumberOfRemainingPlayers = AgentsList.Count;

        //Random platform rot
        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        //Reset Agents
        foreach (var item in AgentsList)
        {
            Debug.Log(item.Agent);
            var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
            var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;

            item.Agent.transform.SetPositionAndRotation(pos, rot);
            item.Rb.velocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
            item.Agent.MyKey.SetActive(false);
            item.Agent.IHaveAKey = false;
            item.Agent.gameObject.SetActive(true);
            m_AgentGroup.RegisterAgent(item.Agent);
        }

        //Reset Key
        // Key.SetActive(false);

        //Reset Tombstone
        //Tombstone.SetActive(false);

        //End Episode
        foreach (var item in DragonsList)
        {
            //var pos = item.StartingPos;
            //var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
            // var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;

            //item.Agent.transform.SetPositionAndRotation(pos, rot);

            item.Agent.gameObject.SetActive(true);
            item.T.SetPositionAndRotation(item.StartingPos, item.StartingRot);
            item.Agent.Initialize();
        }
        Debug.Log("DONEINIT");
    }
}
