using System;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;
using MLAgents.SideChannels;

public class CubeWarSettings : MonoBehaviour
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public CubeWarArea[] listArea;

    public int totalScore;
    public Text scoreText;

    StatsSideChannel m_statsSideChannel;

//    public void Awake()
//    {
//        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
//        m_statsSideChannel = Academy.Instance.GetSideChannel<StatsSideChannel>();
//    }

//    public void EnvironmentReset()
//    {
//
//        agents = GameObject.FindGameObjectsWithTag("agent");
//        listArea = FindObjectsOfType<CubeWarArea>();
//        foreach (var fa in listArea)
//        {
//            fa.ResetWarArea(agents);
//        }
//
//        totalScore = 0;
//    }

//    public void Update()
//    {
//        scoreText.text = $"Score: {totalScore}";
//
//        // Send stats via SideChannel so that they'll appear in TensorBoard.
//        // These values get averaged every summary_frequency steps, so we don't
//        // need to send every Update() call.
//        if ((Time.frameCount % 100)== 0)
//        {
//            m_statsSideChannel?.AddStat("TotalScore", totalScore);
//        }
//    }
}
