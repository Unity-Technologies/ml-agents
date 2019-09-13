using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class BananaAcademy : Academy
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public BananaArea[] listArea;

    public int totalScore;
    public Text scoreText;
    public override void AcademyReset()
    {
        ClearObjects(GameObject.FindGameObjectsWithTag("banana"));
        ClearObjects(GameObject.FindGameObjectsWithTag("badBanana"));

        agents = GameObject.FindGameObjectsWithTag("agent");
        listArea = FindObjectsOfType<BananaArea>();
        foreach (var ba in listArea)
        {
            ba.ResetBananaArea(agents);
        }

        totalScore = 0;
    }

    void ClearObjects(GameObject[] objects)
    {
        foreach (var bana in objects)
        {
            Destroy(bana);
        }
    }

    public override void AcademyStep()
    {
        scoreText.text = string.Format(@"Score: {0}", totalScore);
    }
}
