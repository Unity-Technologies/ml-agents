using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BananaAcademy : Academy
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public List<BananaArea> listArea;

    public int totalScore;
    public Text scoreText;
    public override void AcademyReset()
    {
        ClearObjects(GameObject.FindGameObjectsWithTag("banana"));
        ClearObjects(GameObject.FindGameObjectsWithTag("badBanana"));

        agents = GameObject.FindGameObjectsWithTag("agent");
        foreach (BananaArea ba in listArea)
        {
            ba.ResetArea();
        }

        totalScore = 0;
    }

    void ClearObjects(GameObject[] objects)
    {
        foreach (GameObject bana in objects)
        {
            Destroy(bana);
        }
    }

    public override void AcademyStep()
    {
        scoreText.text = string.Format(@"Score: {0}", totalScore);
    }
}
