using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BouncerAcademy : Academy {

    public float gravityMultiplier = 1f;

    public override void InitializeAcademy()
    {
        Physics.gravity = new Vector3(0,-9.8f*gravityMultiplier,0);
    }

    public override void AcademyReset()
    {


    }

    public override void AcademyStep()
    {


    }

}
