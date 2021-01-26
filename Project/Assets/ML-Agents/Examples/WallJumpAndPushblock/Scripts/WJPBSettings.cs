using UnityEngine;

public class WJPBSettings : MonoBehaviour
{
    [Header("Specific to WallJump")]
    public float agentRunSpeed;
    public float agentJumpHeight;

    [HideInInspector]
    public float agentJumpVelocity = 777;
    [HideInInspector]
    public float agentJumpVelocityMaxChange = 10;

}
