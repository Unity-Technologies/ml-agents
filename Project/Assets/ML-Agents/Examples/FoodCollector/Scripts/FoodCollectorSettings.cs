using System;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class FoodCollectorSettings : MonoBehaviour
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public FoodCollectorArea[] listArea;

    public int totalScore;
    public Text scoreText;

    public int foodEaten;
    public int poisonEaten;

    public void Awake()
    {
        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }

    public void EnvironmentReset()
    {
        ClearObjects(GameObject.FindGameObjectsWithTag("food"));
        ClearObjects(GameObject.FindGameObjectsWithTag("badFood"));

        agents = GameObject.FindGameObjectsWithTag("agent");
        listArea = FindObjectsOfType<FoodCollectorArea>();
        foreach (var fa in listArea)
        {
            fa.ResetFoodArea(agents);
        }

        totalScore = 0;
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
        scoreText.text = $"Score: {totalScore}  f/p:{foodEaten}/{poisonEaten}";
    }

    public void FixedUpdate()
    {
        var ts = Time.time.ToString("f2");
        Debug.Log($"FoodCollectorSettings - FixedUpdate time:{ts} totalScore:{totalScore} Food:{foodEaten} poison:{poisonEaten}");
        Academy.Instance.envStatMan.AddFloatStat("FoodCollector/TotalScore", totalScore);
        Academy.Instance.envStatMan.AddFloatStat("FoodCollector/FoodEaten", foodEaten);
        Academy.Instance.envStatMan.AddFloatStat("FoodCollector/PoisonEaten", poisonEaten);
    }
}
