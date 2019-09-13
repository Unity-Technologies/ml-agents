using UnityEngine;
using MLAgents;

public class HallwayAcademy : Academy
{
    public float agentRunSpeed;
    public float agentRotationSpeed;
    public Material goalScoredMaterial; // when a goal is scored the ground will use this material for a few seconds.
    public Material failMaterial; // when fail, the ground will use this material for a few seconds.
    public float gravityMultiplier; // use ~3 to make things less floaty

    public override void InitializeAcademy()
    {
        Physics.gravity *= gravityMultiplier;
    }

    public override void AcademyReset()
    {
    }
}
