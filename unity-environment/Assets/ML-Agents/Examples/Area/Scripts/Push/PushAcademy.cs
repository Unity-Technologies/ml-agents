using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAcademy : Academy {

    public float objectSize;

	public override void AcademyReset()
	{
        objectSize = (int)resetParameters["object_size"];
	}

	public override void AcademyStep()
	{

	}

}
