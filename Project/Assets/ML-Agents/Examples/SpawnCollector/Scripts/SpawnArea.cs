using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Linq;

public class SpawnArea : MonoBehaviour
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
    public List<SpawnButton> Buttons;
    HashSet<SpawnCollectorFood> Foods = new HashSet<SpawnCollectorFood>();
    int m_NumFoodEaten;

    void Start()
    {
        // Initialize TeamManager
        m_AgentGroup = new SimpleMultiAgentGroup();
        ResetScene();
    }

    public void RegisterFood(SpawnCollectorFood food)
    {
        Foods.Add(food);
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
        foreach (SpawnButton s in Buttons)
        {
            s.ResetSwitch();
        }
        foreach (SpawnCollectorFood f in Foods)
        {
            f.gameObject.SetActive(true);
        }
        m_NumFoodEaten = 0;
        m_ResetTimer = 0;
    }

    public void RegisterAgent(GameObject agent)
    {
        agent.GetComponent<SpawnCollectorAgent>().SetArea(this);
        m_AgentGroup.RegisterAgent(agent.GetComponent<SpawnCollectorAgent>());
    }

    public void FoodEaten()
    {
        m_AgentGroup.AddGroupReward(0.1f);
        m_NumFoodEaten += 1;
        if (m_NumFoodEaten == Foods.Count)
        {
            m_AgentGroup.AddGroupReward(1f);
            m_AgentGroup.EndGroupEpisode();
            ResetScene();
        }
    }

    public float GetNumFoodLeft()
    {
        return 1.0f * (Foods.Count - m_NumFoodEaten);
    }

    public float GetTimeLeft()
    {
        return 1.0f * m_ResetTimer / MaxEnvironmentSteps;
    }

    public void AddReward(float value)
    {
        m_AgentGroup.AddGroupReward(value);
    }

    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            // m_AgentGroup.AddGroupReward(-1f);
            m_AgentGroup.EndGroupEpisode();
            ResetScene();
        }

        //Hurry Up Penalty
        m_AgentGroup.AddGroupReward(-3f / MaxEnvironmentSteps);
    }

    public int GetNumAgents()
    {
        return m_AgentGroup.GetRegisteredAgents().Count;
    }
}
