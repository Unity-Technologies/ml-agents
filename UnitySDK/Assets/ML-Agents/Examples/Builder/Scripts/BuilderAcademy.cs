using UnityEngine;
using MLAgents;

public class BuilderAcademy : Academy
{
    [Header("Specific to WallJump")]
    //when a goal is scored the ground will use this material for a few seconds.
    public Material goalScoredMaterial;
    //when fail, the ground will use this material for a few seconds.
    public Material failMaterial;
    public Material grabbedMaterial;
    public Material notGrabbedMaterial;
    public float heightRewardCoeff = .01f;
}
