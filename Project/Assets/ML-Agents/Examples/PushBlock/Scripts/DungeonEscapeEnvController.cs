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
    public class ZombieInfo
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

    // [System.Serializable]
    // public class BlockInfo
    // {
    //     public Transform T;
    //     [HideInInspector]
    //     public Vector3 StartingPos;
    //     [HideInInspector]
    //     public Quaternion StartingRot;
    //     [HideInInspector]
    //     public Rigidbody Rb;
    //     public Transform LockedBlock;
    // }

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

    // public GameObject area;

    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();
    public List<ZombieInfo> ZombiesList = new List<ZombieInfo>();
    // public List<BlockInfo> BlocksList = new List<BlockInfo>();

    public bool UseRandomAgentRotation = true;
    public bool UseRandomAgentPosition = true;
    // public bool UseRandomBlockRotation = true;
    // public bool UseRandomBlockPosition = true;
    PushBlockSettings m_PushBlockSettings;

    private int m_NumberOfRemainingBlocks;
    public GameObject Key;
    public GameObject LockedBlock;
    // public Rigidbody UnlockedBlock;


    public Dictionary<Transform, PlayerInfo> m_AgentsDict = new Dictionary<Transform, PlayerInfo>();
    public Dictionary<Transform, ZombieInfo> m_ZombiesDict = new Dictionary<Transform, ZombieInfo>();
    // public Dictionary<Transform, BlockInfo> m_BlocksDict = new Dictionary<Transform, BlockInfo>();
    private SimpleMultiAgentGroup m_AgentGroup;
    // public bool BlockIsLocked;
    void Start()
    {

        // Get the ground's bounds
        areaBounds = ground.GetComponent<Collider>().bounds;
        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();

        // //Lock The Block
        // LockTheBlock();

        //Hide The Key
        Key.SetActive(false);

        // foreach (var item in BlocksList)
        // {
        //     item.StartingPos = item.T.transform.position;
        //     item.StartingRot = item.T.transform.rotation;
        //     item.Rb = item.T.GetComponent<Rigidbody>();
        //     m_BlocksDict.Add(item.T, item);
        // }

        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            item.Col = item.Agent.GetComponent<Collider>();
            m_AgentsDict.Add(item.Agent.transform, item);
            // Add to team manager
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        foreach (var item in ZombiesList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.T = item.Agent.transform;
            item.Col = item.Agent.GetComponent<Collider>();
            m_ZombiesDict.Add(item.T, item);
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

        //Hurry Up Penalty
        // m_AgentGroup.AddGroupReward(-0.5f / MaxEnvironmentSteps);
    }

    // // public Dictionary<Agent>
    // //Kill/disable an agent
    // public void KillAgent(Collision col, Transform t)
    // {
    //     print($"zombie {t.name} ate {col.collider.name}");
    //     //Disable killed Agent
    //     foreach (var item in AgentsList)
    //     {
    //         if (item.Col == col.collider)
    //         {
    //             item.Agent.EndEpisode();
    //             item.Col.gameObject.SetActive(false);
    //             break;
    //         }
    //     }
    //
    //     //End Episode
    //     foreach (var item in ZombiesList)
    //     {
    //         if (item.Agent.transform == t)
    //         {
    //             KillZombie(item);
    //             break;
    //         }
    //     }
    // }

    public void UnlockBlock(Transform blockT)
    {
        // LockedBlock.SetActive(false);
        // UnlockedBlock.velocity = Vector3.zero;
        // UnlockedBlock.angularVelocity = Vector3.zero;
        // UnlockedBlock.transform.SetPositionAndRotation(blockT.position, blockT.rotation);
        // UnlockedBlock.gameObject.SetActive(true);
        // BlockIsLocked = false;
        m_AgentGroup.AddGroupReward(1f);
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));

        print("Unlocked Door");
        m_AgentGroup.EndGroupEpisode();

        ResetScene();
    }

    // public void LockTheBlock()
    // {
    //     LockedBlock.SetActive(true);
    //     // UnlockedBlock.velocity = Vector3.zero;
    //     // UnlockedBlock.angularVelocity = Vector3.zero;
    //     // UnlockedBlock.transform.SetPositionAndRotation(LockedBlock.transform.position, LockedBlock.transform.rotation);
    //     // UnlockedBlock.gameObject.SetActive(false);
    //     BlockIsLocked = true;
    // }

    public void KilledByZombie(PushAgentEscape agent, Collision zombCol)
    {
        zombCol.gameObject.SetActive(false);
        agent.EndEpisode();
        agent.gameObject.SetActive(false);
        print($"zombie {zombCol.gameObject.name} ate {agent.transform.name}");
        //Spawn the Key Pickup
        Key.transform.SetPositionAndRotation(zombCol.collider.transform.position, zombCol.collider.transform.rotation);
        Key.SetActive(true);
        // Instantiate(KeyPrefab, zombCol.collider.transform.position, zombCol.collider.transform.rotation, transform);
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

    /// <summary>
    /// Called when the agent moves the block into the goal.
    /// </summary>
    public void ScoredAGoal(Collider col, float score)
    {
        print($"Scored {score} on {gameObject.name}");

        //Give Agent Rewards
        m_AgentGroup.AddGroupReward(score);
        // m_AgentGroup.EndGroupEpisode();

        // Swap ground material for a bit to indicate we scored.
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));
        ResetScene();
    }

    public void ZombieTouchedBlock()
    {
        //Give Agents Penalties
        // m_AgentGroup.AddGroupReward(-1);
        m_AgentGroup.EndGroupEpisode();


        // Swap ground material for a bit to indicate we scored.
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.failMaterial, 0.5f));
        ResetScene();

    }

    Quaternion GetRandomRot()
    {
        return Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
    }

    void ResetScene()
    {

        //Reset counter
        m_ResetTimer = 0;

        //Random platform rot
        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        //Reset Agents
        foreach (var item in AgentsList)
        {
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

        //Reset Blocks
        // LockedBlock.transform.position = GetRandomSpawnPos();
        // LockedBlock.transform.rotation = GetRandomRot();
        // LockTheBlock();

        //Reset Key
        Key.SetActive(false);

        //End Episode
        foreach (var item in ZombiesList)
        {
            if (!item.Agent)
            {
                return;
            }
            // item.Agent.EndEpisode();
            item.Agent.transform.SetPositionAndRotation(item.StartingPos, item.StartingRot);
            item.Agent.SetRandomWalkSpeed();
            item.Agent.gameObject.SetActive(true);
        }



    }

    // void ResetScene()
    // {
    //     m_ResetTimer = 0;
    //
    //     //Random platform rot
    //     var rotation = Random.Range(0, 4);
    //     var rotationAngle = rotation * 90f;
    //     transform.Rotate(new Vector3(0f, rotationAngle, 0f));
    //
    //     //End Episode
    //     foreach (var item in AgentsList)
    //     {
    //         if (!item.Agent)
    //         {
    //             return;
    //         }
    //         item.Agent.EndEpisode();
    //     }
    //     //Reset Agents
    //     foreach (var item in AgentsList)
    //     {
    //         var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
    //         var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;
    //
    //         item.Agent.transform.SetPositionAndRotation(pos, rot);
    //         item.Rb.velocity = Vector3.zero;
    //         item.Rb.angularVelocity = Vector3.zero;
    //         item.Agent.gameObject.SetActive(true);
    //     }
    //
    //     //Reset Blocks
    //     foreach (var item in BlocksList)
    //     {
    //         var pos = UseRandomBlockPosition ? GetRandomSpawnPos() : item.StartingPos;
    //         var rot = UseRandomBlockRotation ? GetRandomRot() : item.StartingRot;
    //
    //         item.T.transform.SetPositionAndRotation(pos, rot);
    //         item.Rb.velocity = Vector3.zero;
    //         item.Rb.angularVelocity = Vector3.zero;
    //         item.T.gameObject.SetActive(true);
    //         // BlockIsLocked(item, true);
    //     }
    //     //End Episode
    //     foreach (var item in ZombiesList)
    //     {
    //         if (!item.Agent)
    //         {
    //             return;
    //         }
    //         // item.Agent.EndEpisode();
    //         item.Agent.transform.SetPositionAndRotation(item.StartingPos, item.StartingRot);
    //         item.Agent.SetRandomWalkSpeed();
    //         item.Agent.gameObject.SetActive(true);
    //     }
    //
    //     //Reset counter
    //     m_NumberOfRemainingBlocks = BlocksList.Count;
    //     // m_NumberOfRemainingBlocks = 2;
    // }
}
