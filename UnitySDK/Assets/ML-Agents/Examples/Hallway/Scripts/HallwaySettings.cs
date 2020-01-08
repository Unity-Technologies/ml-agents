using UnityEngine;
using MLAgents;

public class HallwaySettings : Academy
{
    public float agentRunSpeed;
    public float agentRotationSpeed;
    public Material goalScoredMaterial; // when a goal is scored the ground will use this material for a few seconds.
    public Material failMaterial; // when fail, the ground will use this material for a few seconds.

    public override void AcademyReset()
    {
    }
}
