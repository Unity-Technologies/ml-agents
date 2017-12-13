using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaAcademy : Academy {

	public override void AcademyReset()
	{
        GameObject.Find("AreaPB").GetComponent<BananaArea>().ResetArea();
	}

	public override void AcademyStep()
	{

	}

}
