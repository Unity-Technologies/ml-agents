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
        ClearObjects(GameObject.FindGameObjectsWithTag("banana"));
        ClearObjects(GameObject.FindGameObjectsWithTag("badBanana"));

		agents = GameObject.FindGameObjectsWithTag("agent");
        foreach ( BananaArea ba in listArea){
            ba.ResetArea();
        }

	}

    void ClearObjects(GameObject[] objects) {
        foreach (GameObject bana in objects)
        {
            Destroy(bana);
        }
    }

	public override void AcademyStep()
	{

	}

}
