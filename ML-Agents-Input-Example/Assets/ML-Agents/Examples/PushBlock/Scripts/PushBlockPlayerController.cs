using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;

public class PushBlockPlayerController : MonoBehaviour, IInputActionAssetProvider
{

    PushBlockSettings m_PushBlockSettings;
    public float JumpTime = 0.5f;
    float m_JumpTimeRemaining;
    Rigidbody m_PlayerRb;  //cached on initialization
    PushBlockActions m_PushBlockActions;
    float m_JumpCoolDownStart;

    void Awake()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        LazyInitializeActions();

        // Cache the agent rigidbody
        m_PlayerRb = GetComponent<Rigidbody>();
    }

    void LazyInitializeActions()
    {
        if (m_PushBlockActions != null)
        {
            return;
        }
        m_PushBlockActions = new PushBlockActions();
        m_PushBlockActions.Enable();
        m_PushBlockActions.Movement.jump.performed += JumpOnperformed;
    }

    void JumpOnperformed(InputAction.CallbackContext callbackContext)
    {
        InnerJump(gameObject.transform);
    }

    void FixedUpdate()
    {
        InnerMove(gameObject.transform, m_PushBlockActions.Movement.movement.ReadValue<Vector2>());
        if (m_JumpTimeRemaining < 0)
        {
            m_PlayerRb.AddForce(-transform.up * (m_PushBlockSettings.agentJumpForce * 3), ForceMode.Acceleration);
        }
        m_JumpTimeRemaining -= Time.fixedDeltaTime;
    }

    void InnerJump(Transform t)
    {
        if (Time.realtimeSinceStartup - m_JumpCoolDownStart > m_PushBlockSettings.agentJumpCoolDown)
        {
            m_JumpTimeRemaining = JumpTime;
            m_PlayerRb.AddForce(t.up * m_PushBlockSettings.agentJumpForce, ForceMode.VelocityChange);
            m_JumpCoolDownStart = Time.realtimeSinceStartup;
        }
    }

    void InnerMove(Transform t, Vector2 v)
    {
        var forward = CreateForwardVector(v);
        var up = CreateUpVector(v);
        var dirToGo = t.forward * forward;
        var rotateDir = t.up * up;
        t.Rotate(rotateDir, Time.deltaTime * 200f);
        m_PlayerRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    static float CreateUpVector(Vector2 move)
    {
        return Mathf.Abs(move.x) > Mathf.Abs(move.y) ? move.x : 0f;
    }

    static float CreateForwardVector(Vector2 move)
    {
        return Mathf.Abs(move.y) > Mathf.Abs(move.x) ? move.y : 0f;
    }

    public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
    {
        LazyInitializeActions();
        return (m_PushBlockActions.asset, m_PushBlockActions);
    }
}
