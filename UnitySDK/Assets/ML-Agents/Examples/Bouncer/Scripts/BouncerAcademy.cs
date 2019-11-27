using UnityEngine;
using MLAgents;

public class BouncerAcademy : Academy
{
    public float gravityMultiplier = 1f;

    public override void InitializeAcademy()
    {
        FloatProperties.RegisterCallback("target_scale", f => { });
        Physics.gravity = new Vector3(0, -9.8f * gravityMultiplier, 0);
    }

    public override void AcademyReset()
    {
    }

    public override void AcademyStep()
    {
    }
}
