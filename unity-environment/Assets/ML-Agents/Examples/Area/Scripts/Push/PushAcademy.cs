using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAcademy : Academy {

    public float objectSize;

	public override void AcademyReset()
	{
        int steps = (int)resetParameters["steps"];
        setCurriculum(steps);
	}

	public override void AcademyStep()
	{

	}

    public void setCurriculum(int steps) {
        if (steps < 20000) {
            objectSize = 2;
        }
        else if (steps > 20000 && steps < 40000) {
            objectSize = 1.8f;
        }
        else if (steps > 40000 && steps < 80000) {
            objectSize = 1.6f;
        }
        else if (steps > 80000 && steps < 160000) {
            objectSize = 1.4f;
        }
        else if (steps > 160000 && steps < 320000) 
        {
            objectSize = 1.2f;
		}
        else {
            objectSize = 1.0f;
		}
    }

}
