using UnityEngine;
using MLAgents;

public class WallJumpAcademy : Academy
{
    [Header("Specific to WallJump")]
    public float agentRunSpeed;
    public float agentJumpHeight;
    //when a goal is scored the ground will use this material for a few seconds.
    public Material goalScoredMaterial;
    //when fail, the ground will use this material for a few seconds.
    public Material failMaterial;

    [HideInInspector]
    //use ~3 to make things less floaty
    public float gravityMultiplier = 2.5f;
    [HideInInspector]
    public float agentJumpVelocity = 777;
    [HideInInspector]
    public float agentJumpVelocityMaxChange = 10;

    // Use this for initialization
    public override void InitializeAcademy()
    {
        Physics.gravity *= gravityMultiplier;
    }
}
