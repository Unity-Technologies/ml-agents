//Standardized movement controller for the Agent Cube
using UnityEngine;

public class TennisRacketMovement : MonoBehaviour
{
    [Header("RUNNING")] public ForceMode runningForceMode = ForceMode.Impulse;
    //speed agent can run if grounded
    public float agentRunSpeed = 20; 
    //speed agent can run if not grounded
    public float agentRunInAirSpeed = 10f; 
    
    [Header("IDLE")]
    //coefficient used to dampen velocity when idle
    //the purpose of this is to fine tune agent drag
    //...and prevent the agent sliding around while grounded
    //0 means it will instantly stop when grounded
    //1 means no drag will be applied
    public float agentIdleDragVelCoeff = .85f; 
    
    [Header("BODY ROTATION")]
    //body rotation speed
    public float agentRotationSpeed = 7f;

    [Header("JUMPING")]
    //upward jump velocity magnitude 
    public float agentJumpVelocity = 15f;

    [Header("FALLING FORCE")]
    //force applied to agent while falling
    public float agentFallingSpeed = 50f;

    public void Jump(Rigidbody rb)
    {
        Vector3 velToUse = rb.velocity;
        velToUse.y = agentJumpVelocity;
        rb.velocity = velToUse;
    }
    
    public void RotateBody(Rigidbody rb, Vector3 rotationAxis)
    {
        rb.MoveRotation(rb.rotation * Quaternion.AngleAxis(agentRotationSpeed, rotationAxis));
    }

    public void RunOnGround(Rigidbody rb, Vector3 dir)
    {
        var vel = rb.velocity.magnitude;
        float adjustedSpeed = Mathf.Clamp(agentRunSpeed - vel, 0, agentRunSpeed);
        rb.AddForce(dir.normalized * adjustedSpeed,
            runningForceMode); 
    }
    
    public void RunInAir(Rigidbody rb, Vector3 dir)
    {
        var vel = rb.velocity.magnitude;
        float adjustedSpeed = Mathf.Clamp(agentRunInAirSpeed - vel, 0, agentRunInAirSpeed);
        rb.AddForce(dir.normalized * adjustedSpeed,
            runningForceMode); 
    }

    public void AddIdleDrag(Rigidbody rb)
    {
        rb.velocity *= agentIdleDragVelCoeff;
    }
    
    public void AddFallingForce(Rigidbody rb)
    {
        rb.AddForce(
            Vector3.down * agentFallingSpeed, ForceMode.Acceleration);
    }
}
    
