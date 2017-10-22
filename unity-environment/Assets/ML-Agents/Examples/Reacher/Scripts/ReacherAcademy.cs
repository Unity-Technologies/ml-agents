using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReacherAcademy : Academy {

    public float goalSize;

	public override void AcademyReset()
	{
        goalSize = (float)resetParameters["goal_size"];
	}

	public override void AcademyStep()
	{


	}

}
