using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// This class handles the input for the PushBlock Cube character in the PushBlock scene.
/// Note that the only ML-Agents code here is the implementation of the <see cref="IInputActionAssetProvider"/>.
/// The <see cref="InputActuatorComponent"/> looks for a component that implements that interface in order to
/// rebind actions to virtual controllers when training agents or running inference.  This means that you can
/// keep your input handling code separate from ML-Agents, and have your agent's action space defined by the
/// actions defined in your project's <see cref="GetInputActionAsset"/>.
///
/// If you don't implement <see cref="IInputActionAssetProvider"/> the <see cref="InputActuatorComponent"/> will
/// look for a <see cref="PlayerInput"/> component on the GameObject it live on.  It will rebind the actions of that
/// instance of the asset.
///
/// It is important to note that if you have multiple components on the same GameObject handling input, you will
/// need to share the instance of the generated C# <see cref="IInputActionCollection2"/> (named <see cref="m_PushBlockActions"/>
/// here) in order to ensure that all of your actions are bound correctly for ml-agents training and inference.
/// </summary>
public class PushBlockWithInputPlayerController : MonoBehaviour, IInputActionAssetProvider
{

    PushBlockWithInputSettings m_PushBlockSettings;
    public float JumpTime = 0.5f;
    float m_JumpTimeRemaining;
    Rigidbody m_PlayerRb; //cached on initialization
    PushBlockActions m_PushBlockActions;
    float m_JumpCoolDownStart;

    void Awake()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockWithInputSettings>();
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

        // You can listen to C# events.
        m_PushBlockActions.Movement.jump.performed += JumpOnperformed;
    }

    void JumpOnperformed(InputAction.CallbackContext callbackContext)
    {
        InnerJump(gameObject.transform);
    }

    void FixedUpdate()
    {
        // Or you can poll the action itself like we do here.
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

    /// <summary>
    /// This is the implementation of the <see cref="IInputActionAssetProvider"/> for this class.  We need
    /// both the <see cref="GetInputActionAsset"/> and the <see cref="IInputActionCollection2"/> if you are
    /// listening to C# events, Unity Events, or receiving Messages from the Input System Package as those callbacks
    /// are set up through the generated <see cref="IInputActionCollection2"/>.
    /// </summary>
    /// <returns></returns>
    public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
    {
        LazyInitializeActions();
        return (m_PushBlockActions.asset, m_PushBlockActions);
    }
}
