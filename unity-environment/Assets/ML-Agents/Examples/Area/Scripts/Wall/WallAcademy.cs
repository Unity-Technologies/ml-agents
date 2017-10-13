using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallAcademy : Academy {

    public int minWallHeght;
    public int maxWallHeight;

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
            minWallHeght = 1;
            maxWallHeight = 3;
        }
        else if (steps > 20000 && steps < 40000) {
            minWallHeght = 1;
            maxWallHeight = 4;
        }
        else if (steps > 40000 && steps < 80000) {
            minWallHeght = 2;
            maxWallHeight = 4;
        }
        else if (steps > 80000 && steps < 160000) {
            minWallHeght = 2;
            maxWallHeight = 5;
        }
        else if (steps > 160000 && steps < 320000) 
        {
			minWallHeght = 3;
			maxWallHeight = 5;
		}
        else {
			minWallHeght = 4;
			maxWallHeight = 5;
		}
    }

}
