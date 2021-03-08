using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class EscapeRoomGameController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public EscapeRoomAgentCollab Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
        public bool Scored;
        public bool HoldingSwitch;

    }

    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;

    public int NumOfAgentsThatHaveScored = 0;
    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();

    public Dictionary<EscapeRoomAgentCollab, PlayerInfo> PlayerDict =
        new Dictionary<EscapeRoomAgentCollab, PlayerInfo>();
    private SimpleMultiAgentGroup m_AgentGroup;
    private int m_ResetTimer;
    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    public MeshRenderer GroundRenderer;
    public Material GroundMaterial; //cached on Awake()
    public Material GoalScoredMaterial;
    // Start is called before the first frame update
    void Start()
    {
        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.localPosition;
            item.StartingRot = item.Agent.transform.localRotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            m_AgentGroup.RegisterAgent(item.Agent);
            PlayerDict.Add(item.Agent, item);
        }
        ResetScene();

    }


    bool AgentIsHoldingSwitch()
    {
        foreach (var item in PlayerDict)
        {
            if (item.Value.HoldingSwitch)
            {
                return true;
            }
        }
        return false;
    }

    public bool DoorIsOpen;
    // Update is called once per frame
    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }

        DoorIsOpen = AgentIsHoldingSwitch();
        DoorOpen(DoorIsOpen);
        foreach (var item in PlayerDict)
        {
            item.Value.HoldingSwitch = false;
        }
    }

    public GameObject Door;
    void DoorOpen(bool isOpen)
    {
        Door.SetActive(!isOpen);
    }

    void ResetScene()
    {
        NumOfAgentsThatHaveScored = 0;
        m_ResetTimer = 0;


        print($"Resetting {gameObject.name}");
        foreach (var item in AgentsList)
        {
            item.Agent.transform.localPosition = item.StartingPos;
            item.Agent.transform.localRotation = item.StartingRot;
            item.Rb.velocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
            item.Scored = false;
            item.HoldingSwitch = false;
        }

        DoorIsOpen = false;
        DoorOpen(false);
        //Random platform rot
        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        transform.Rotate(new Vector3(0f, rotationAngle, 0f));
    }

    public void AgentScored(EscapeRoomAgentCollab agent)
    {
        if (PlayerDict[agent].Scored)
        {
            return;
        }
        print($"{agent.name} Scored");
        PlayerDict[agent].Scored = true;
        //Decrement the counter
        NumOfAgentsThatHaveScored++;

        // Swap ground material for a bit to indicate we scored.
        StartCoroutine(GoalScoredSwapGroundMaterial(GoalScoredMaterial, 0.3f));

        //Give Agent Rewards
        m_AgentGroup.AddGroupReward(1);

        if (NumOfAgentsThatHaveScored == 3)
        {
            //Reset assets
            m_AgentGroup.EndGroupEpisode();
            ResetScene();
        }
    }

    /// <summary>
    /// Swap ground material, wait time seconds, then swap back to the regular material.
    /// </summary>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        print("Start GoalScoredSwapGroundMaterial");
        GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        GroundRenderer.material = GroundMaterial;
    }
}
