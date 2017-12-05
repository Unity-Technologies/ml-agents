using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAcademy : Academy {

    public float goalSize;
    public float blockSize;
    public float xVariation;

	public override void AcademyReset()
	{
        goalSize = (float)resetParameters["goal_size"];
        blockSize = (float)resetParameters["block_size"];
        xVariation = (float)resetParameters["x_variation"];
	}

	public override void AcademyStep()
	{

	}

}
