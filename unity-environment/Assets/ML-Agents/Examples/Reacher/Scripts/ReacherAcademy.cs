using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReacherAcademy : Academy {

    public float goalSize;
    public float goalSpeed;


    public override void AcademyReset()
    {
        goalSize = resetParameters["goal_size"];
        goalSpeed = resetParameters["goal_speed"];
    }

    public override void AcademyStep()
    {


    }

}
