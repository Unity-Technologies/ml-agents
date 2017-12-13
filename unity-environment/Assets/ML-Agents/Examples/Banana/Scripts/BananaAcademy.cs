using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaAcademy : Academy {
    [HideInInspector]
    public GameObject[] agents;
	[HideInInspector]
    public List<BananaArea> listArea;
	public override void AcademyReset()
	{
		GameObject[] oldBanans = GameObject.FindGameObjectsWithTag("banana");
		foreach (GameObject bana in oldBanans)
		{
			Destroy(bana);
		}
		agents = GameObject.FindGameObjectsWithTag("agent");
        foreach ( BananaArea ba in listArea){
            ba.ResetArea();
        }

	}

	public override void AcademyStep()
	{

	}

}
