using UnityEngine;
using MLAgents;

public class TennisAcademy : Academy
{
    public override void AcademyReset()
    {
        Physics.gravity = new Vector3(0, -resetParameters["gravity"], 0);
    }

    public override void AcademyStep()
    {
    }
}
