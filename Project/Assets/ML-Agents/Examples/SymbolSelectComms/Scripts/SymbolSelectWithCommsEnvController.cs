using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class SymbolSelectWithCommsEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public SymboSelectWithComms Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
    }

    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;

    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    //List of Agents On Platform
    public PlayerInfo[] AgentsArray = new PlayerInfo[0];
    private PushBlockSettings m_PushBlockSettings;
    private SimpleMultiAgentGroup m_AgentGroup;
    void Start()
    {

        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        foreach (var item in AgentsArray)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        // ResetScene();
    }

    void FixedUpdate()
    {
        var done = true;
        foreach (var item in AgentsArray)
        {
            if (!item.Agent.hasMovedToTarget)
            {
                done = false;
                break;
            }
        }

        if (done)
        {

            var correctAnswer = AgentsArray[0].Agent.assignedNumber ^ AgentsArray[1].Agent.assignedNumber;

            // //doublecheck xor math in console
            // print($"Assigned - 0:{AgentsArray[0].Agent.assignedNumber} 1:{AgentsArray[1].Agent.assignedNumber} correct: {correctAnswer}");

            //IF SECOND AGENT CHOSE THE SAME AS THE FIRST GIVE IT A REWARD
            if (AgentsArray[0].Agent.answerChoice == AgentsArray[1].Agent.assignedNumber && AgentsArray[1].Agent.answerChoice == AgentsArray[0].Agent.assignedNumber)
            {
            // print(
            //     $"Chose Correctly - 0:{AgentsArray[0].Agent.answerChoice} 1:{AgentsArray[1].Agent.answerChoice} correct: {correctAnswer}");
                m_AgentGroup.AddGroupReward(1f);
            }
            else
            {
                m_AgentGroup.AddGroupReward(-1f);
            // print(
            //     $"Chose Incorrectly - 0:{AgentsArray[0].Agent.answerChoice} 1:{AgentsArray[1].Agent.answerChoice} correct: {correctAnswer}");
            }
            m_AgentGroup.EndGroupEpisode();
            // ResetScene();
        }

        // if (done)
        // {
        //
        //     var correctAnswer = AgentsArray[0].Agent.assignedNumber ^ AgentsArray[1].Agent.assignedNumber;
        //
        //     // //doublecheck xor math in console
        //     // print($"Assigned - 0:{AgentsArray[0].Agent.assignedNumber} 1:{AgentsArray[1].Agent.assignedNumber} correct: {correctAnswer}");
        //
        //     //IF SECOND AGENT CHOSE THE SAME AS THE FIRST GIVE IT A REWARD
        //     if (AgentsArray[0].Agent.answerChoice == correctAnswer && AgentsArray[1].Agent.answerChoice == correctAnswer)
        //     {
        //     // print(
        //     //     $"Chose Correctly - 0:{AgentsArray[0].Agent.answerChoice} 1:{AgentsArray[1].Agent.answerChoice} correct: {correctAnswer}");
        //         m_AgentGroup.AddGroupReward(1f);
        //     }
        //     else
        //     {
        //         m_AgentGroup.AddGroupReward(-1f);
        //     // print(
        //     //     $"Chose Incorrectly - 0:{AgentsArray[0].Agent.answerChoice} 1:{AgentsArray[1].Agent.answerChoice} correct: {correctAnswer}");
        //     }
        //     m_AgentGroup.EndGroupEpisode();
        //     // ResetScene();
        // }
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

    // /// <summary>
    // /// Called when the agent moves the block into the goal.
    // /// </summary>
    // public void ScoredAGoal(Collider col, float score)
    // {
    //     // if (expr)
    //     // {
    //     //
    //     // }
    //     print($"Scored {score} on {gameObject.name}");
    //
    //     //Decrement the counter
    //     m_NumberOfRemainingBlocks--;
    //
    //     //Are we done?
    //     bool done = m_NumberOfRemainingBlocks == 0;
    //
    //     //Disable the block
    //     col.gameObject.SetActive(false);
    //
    //     //Give Agent Rewards
    //     m_AgentGroup.AddGroupReward(score);
    //
    //     // Swap ground material for a bit to indicate we scored.
    //     StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));
    //
    //     if (done)
    //     {
    //         //Reset assets
    //         m_AgentGroup.EndGroupEpisode();
    //         ResetScene();
    //     }
    // }

    // Quaternion GetRandomRot()
    // {
    //     return Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
    // }

    // void ResetAgent(PlayerInfo item, bool isFirstAgent)
    // {
    //     var pos = item.StartingPos;
    //     var rot = item.StartingRot;
    //
    //     item.Agent.transform.SetPositionAndRotation(pos, rot);
    //     item.Rb.velocity = Vector3.zero;
    //     item.Rb.angularVelocity = Vector3.zero;
    //     if (isFirstAgent)
    //     {
    //         item.Agent.canChooseNow = true;
    //     }
    //     else
    //     {
    //         item.Agent.canChooseNow = false;
    //     }
    //     item.Agent.hasChosen = false;
    //     item.Agent.hasMovedToTarget = false;
    // }
    //
    // public void ResetScene()
    // {
    //     //Reset Agents
    //     ResetAgent(AgentsArray[0], true);
    //     ResetAgent(AgentsArray[1], false);
    // }
}
