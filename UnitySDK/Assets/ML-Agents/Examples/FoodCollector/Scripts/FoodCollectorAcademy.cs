using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class FoodCollectorAcademy : Academy
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public FoodCollectorArea[] listArea;

    public int totalScore;
    public Text scoreText;
    public override void AcademyReset()
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

    public override void InitializeAcademy()
    {
        FloatProperties.RegisterCallback("laser_length", f => { });
        FloatProperties.RegisterCallback("agent_scale", f => { });
    }

    void ClearObjects(GameObject[] objects)
    {
        foreach (var food in objects)
        {
            Destroy(food);
        }
    }

    public override void AcademyStep()
    {
        scoreText.text = string.Format(@"Score: {0}", totalScore);
    }
}
