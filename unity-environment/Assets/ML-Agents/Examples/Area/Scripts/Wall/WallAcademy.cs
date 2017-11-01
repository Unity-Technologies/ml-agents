using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAcademy : Academy {

    public int minWallHeight;
    public int maxWallHeight;

	public override void AcademyReset()
	{
        minWallHeight = (int)resetParameters["min_wall_height"];
        maxWallHeight = (int)resetParameters["max_wall_height"];
	}

	public override void AcademyStep()
	{

    }

}
