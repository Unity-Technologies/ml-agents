using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class TennisAcademy : Academy
{

    public override void AcademyReset()
    {
        Debug.Log("Learn WR: " + TennisCanvas.learnWinrate());
        Debug.Log("Blue WR: " + TennisCanvas.blueWinrate());
        Debug.Log("Average Passes: " + TennisCanvas.averagePasses());
    }

    public override void AcademyStep()
    {
    }

}
