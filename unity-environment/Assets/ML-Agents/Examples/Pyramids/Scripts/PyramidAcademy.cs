using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PyramidAcademy : Academy
{
    [HideInInspector]
    public GameObject[] agents;
    [HideInInspector]
    public PyramidArea[] listArea;

    public override void AcademyReset()
    {
        agents = GameObject.FindGameObjectsWithTag("agent");
        listArea = FindObjectsOfType<PyramidArea>();
        foreach (PyramidArea ba in listArea)
        {
            ba.ResetPyramidArea(agents);
        }
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

    }
}
