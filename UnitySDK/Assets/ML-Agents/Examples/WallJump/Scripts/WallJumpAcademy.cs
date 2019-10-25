using UnityEngine;
using MLAgents;

public class WallJumpAcademy : Academy
{
    [Header("Specific to WallJump")]
    //when a goal is scored the ground will use this material for a few seconds.
    public Material goalScoredMaterial;
    //when fail, the ground will use this material for a few seconds.
    public Material failMaterial;
}
