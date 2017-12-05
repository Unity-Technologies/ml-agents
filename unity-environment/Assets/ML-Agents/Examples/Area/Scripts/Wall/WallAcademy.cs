using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAcademy : Academy {

    public float minWallHeight;
    public float maxWallHeight;

	public override void AcademyReset()
	{
        minWallHeight = (float)resetParameters["min_wall_height"];
        maxWallHeight = (float)resetParameters["max_wall_height"];
	}

	public override void AcademyStep()
	{

    }

}
