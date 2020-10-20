using UnityEngine;
using UnityEngine.UI;
using Unity.MLAgents;
using System.Collections;
using System.Collections.Generic;


[System.Serializable]
public class BattleAgentState
{
    public int teamId;
    public BattleFoodAgent agent;
}

public class BattleFoodScoring : MonoBehaviour
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public FoodCollectorArea[] listArea;

    public int WinningScore;
    public Text scoreText;
    public List<BattleAgentState> playerStates = new List<BattleAgentState>();

    StatsRecorder m_Recorder;
    Dictionary<int, int> m_Scores;

    public void Awake()
    {
        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
        m_Recorder = Academy.Instance.StatsRecorder;
        m_Scores = new Dictionary<int, int>();
    }

    void EnvironmentReset()
    {
        ClearObjects(GameObject.FindGameObjectsWithTag("food"));
        ClearObjects(GameObject.FindGameObjectsWithTag("badFood"));

        agents = GameObject.FindGameObjectsWithTag("agent");
        listArea = FindObjectsOfType<FoodCollectorArea>();
        foreach (var fa in listArea)
        {
            fa.ResetFoodArea(agents);
        }
        m_Scores.Clear();
    }

    void ClearObjects(GameObject[] objects)
    {
        foreach (var food in objects)
        {
            Destroy(food);
        }
    }

    public void Update()
    {

        // Send stats via SideChannel so that they'll appear in TensorBoard.
        // These values get averaged every summary_frequency steps, so we don't
        // need to send every Update() call.
        if ((Time.frameCount % 100) == 0)
        {
            foreach (KeyValuePair<int, int> entry in m_Scores)
            {
                m_Recorder.Add($"Team {entry.Key} Score", entry.Value);
            }
        }
    }

    public void AddScore(int teamId, int score)
    {
        if (m_Scores.ContainsKey(teamId))
        {
            m_Scores[teamId] += score;
        }
        else
        {
            m_Scores[teamId] = score;
        }
        if (m_Scores[teamId] >= WinningScore)
        {
            foreach (BattleAgentState state in playerStates)
            {
                if (state.teamId == teamId)
                {
                    state.agent.AddReward(1);
                }
                else
                {
                    state.agent.AddReward(-1);
                }
                state.agent.EndEpisode();
            }
            m_Scores.Clear();
        }
    }

    public int GetScore(int teamId)
    {
        if (m_Scores.ContainsKey(teamId))
        {
            return m_Scores[teamId];
        }
        else
        {
            m_Scores[teamId] = 0;
            return 0;
        }
    }
}
