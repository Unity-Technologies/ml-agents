using UnityEngine;
using MLAgents;

public class TennisAcademy : Academy
{
    public override void InitializeAcademy()
    {
        FloatProperties.RegisterCallback("gravity", f => { Physics.gravity = new Vector3(0, -f, 0); });

    }

    public override void AcademyStep()
    {
    }
}
