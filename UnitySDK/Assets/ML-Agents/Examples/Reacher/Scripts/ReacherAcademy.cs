using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ReacherAcademy : Academy {

    public float goalSize;
    public float goalSpeed;


    public override void AcademyReset()
    {
        goalSize = (float)resetParameters["goal_size"];
        goalSpeed = (float)resetParameters["goal_speed"];
    }

    public override void AcademyStep()
    {


    }

}
