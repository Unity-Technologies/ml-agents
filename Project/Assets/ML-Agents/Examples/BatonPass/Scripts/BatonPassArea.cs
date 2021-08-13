using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Linq;

public class BatonPassArea : MonoBehaviour
{
    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 10000;

    public int m_ResetTimer;

    private SimpleMultiAgentGroup m_AgentGroup;
    public GameObject AgentSpawnPosition;
    public GameObject AgentPrefab;
    public BatonPassButton Button;
    public BatonPassFood Food;
    public int MaxFood;
    int NUMAGENT;
    float m_CumulativeGroupReward = 0f;
    int m_NumFoodEaten = 0;

    void Start()
    {
        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        ResetScene();
    }

    void ResetScene()
    {
        var agents = m_AgentGroup.GetRegisteredAgents().ToList();
        foreach (Agent a in agents)
        {
            m_AgentGroup.UnregisterAgent(a);
            Destroy(a.gameObject);
        }
        var firstAgent = Instantiate(AgentPrefab, AgentSpawnPosition.transform.position, default(Quaternion), gameObject.transform);
        RegisterAgent(firstAgent);
        NUMAGENT = 1;

        Button.ResetSwitch();

        Food.gameObject.SetActive(false);
        m_ResetTimer = 0;
        m_CumulativeGroupReward = 0.0f;
        m_NumFoodEaten = 0;
    }

    public void RegisterAgent(GameObject agent)
    {
        NUMAGENT += 1;
        agent.GetComponent<BatonPassAgent>().SetArea(this);
        m_AgentGroup.RegisterAgent(agent.GetComponent<BatonPassAgent>());
    }

    public void UnregisterAgent(GameObject agent, bool removeFromGroup)
    {
        NUMAGENT -= 1;
        if (removeFromGroup)
        {
            m_AgentGroup.UnregisterAgent(agent.GetComponent<BatonPassAgent>());
        }
    }

    public void FoodEaten()
    {
        Button.ResetSwitch();
        m_AgentGroup.AddGroupReward(1f);
        m_CumulativeGroupReward += 1f;
        m_NumFoodEaten += 1;
        if (m_NumFoodEaten >= MaxFood)
        {
            m_AgentGroup.GroupEpisodeInterrupted();
            Academy.Instance.StatsRecorder.Add("Environment/Actual Group Reward", m_CumulativeGroupReward);
            Academy.Instance.StatsRecorder.Add("FoodEaten", m_NumFoodEaten);
            ResetScene();
        }
    }

    // public void AddReward(float value)
    // {
    //     m_AgentGroup.AddGroupReward(value);
    // }

    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= Academy.Instance.EnvironmentParameters.GetWithDefault("area_steps", MaxEnvironmentSteps) && Academy.Instance.EnvironmentParameters.GetWithDefault("area_steps", MaxEnvironmentSteps) > 0)
        {
            // m_AgentGroup.AddGroupReward(-1f);
            m_AgentGroup.GroupEpisodeInterrupted();
            Academy.Instance.StatsRecorder.Add("Environment/Actual Group Reward", m_CumulativeGroupReward);
            Academy.Instance.StatsRecorder.Add("FoodEaten", m_NumFoodEaten);
            ResetScene();
            return;
        }

        if (NUMAGENT == 0)
        {
            m_AgentGroup.EndGroupEpisode();
            Academy.Instance.StatsRecorder.Add("Environment/Actual Group Reward", m_CumulativeGroupReward);
            Academy.Instance.StatsRecorder.Add("FoodEaten", m_NumFoodEaten);
            ResetScene();
            return;
        }

        //Hurry Up Penalty
        // var penalty = -0.5f / Academy.Instance.EnvironmentParameters.GetWithDefault("area_steps", MaxEnvironmentSteps);
        // m_AgentGroup.AddGroupReward(penalty);
        // m_CumulativeGroupReward += penalty;
    }

    public int GetNumAgents()
    {
        return NUMAGENT;
    }
}
