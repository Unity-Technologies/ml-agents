using System;
using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;
using UnityEngine.UI;


public class TankShooting : MonoBehaviour, IInputActionAssetProvider
{
    [FormerlySerializedAs("m_PlayerNumber")]
    public int playerNumber = 1;                  // Used to identify the different players.
    [FormerlySerializedAs("m_Shell")]
    public Rigidbody shell;                       // Prefab of the shell.
    [FormerlySerializedAs("m_FireTransform")]
    public Transform fireTransform;               // A child of the tank where the shells are spawned.
    [FormerlySerializedAs("m_AimSlider")]
    public Slider aimSlider;                      // A child of the tank that displays the current launch force.
    [FormerlySerializedAs("m_ShootingAudio")]
    public AudioSource shootingAudio;             // Reference to the audio source used to play the shooting audio. NB: different to the movement audio source.
    [FormerlySerializedAs("m_ChargingClip")]
    public AudioClip chargingClip;                // Audio that plays when each shot is charging up.
    [FormerlySerializedAs("m_FireClip")]
    public AudioClip fireClip;                    // Audio that plays when each shot is fired.
    [FormerlySerializedAs("m_MinLaunchForce")]
    public float minLaunchForce = 15f;            // The force given to the shell if the fire button is not held.
    [FormerlySerializedAs("m_MaxLaunchForce")]
    public float maxLaunchForce = 30f;            // The force given to the shell if the fire button is held for the max charge time.
    [FormerlySerializedAs("m_MaxChargeTime")]
    public float maxChargeTime = 0.75f;           // How long the shell can charge for before it is fired at max force.

    internal float m_CurrentLaunchForce;             // The force that will be given to the shell when the fire button is released.
    float m_ChargeSpeed;                    // How fast the launch force increases, based on the max charge time.
    bool m_Fired;                           // Whether or not the shell has been launched with this button press.
    bool m_FireButtonPressedThisFrame;      // Will be set to true when the fire button is initially pressed.
    bool m_FireButtonReleasedThisFrame;     // Will be set to true when the fire button is first released.
    bool m_FireButtonDown;                  // Will always be true while the fire button is held down.
    TankAgent m_TankAgent;
    public TanksInputActions actions;

    void OnEnable()
    {
        // When the tank is turned on, reset the launch force and the UI
        m_CurrentLaunchForce = minLaunchForce;
        aimSlider.value = minLaunchForce;
        m_TankAgent = GetComponent<TankAgent>();
        InitializeAction();
        actions.Player.Fire.performed += CheckFired;
        actions.Player.Fire.started += CheckFired;
        actions.Player.Fire.canceled += CheckFired;
    }

    void OnDisable()
    {
        actions.Player.Fire.performed -= CheckFired;
        actions.Player.Fire.started -= CheckFired;
        actions.Player.Fire.canceled -= CheckFired;
    }

    void CheckFired(InputAction.CallbackContext callbackContext)
    {
        m_FireButtonPressedThisFrame = callbackContext.started;
        m_FireButtonReleasedThisFrame = callbackContext.canceled;
        m_FireButtonDown = callbackContext.performed;

        if (callbackContext.started)
        {
            m_Fired = false;
        }

    }

    void Start()
    {
        // The rate that the launch force charges up is the range of possible forces by the max charge time.
        m_ChargeSpeed = (maxLaunchForce - minLaunchForce) / maxChargeTime;
        actions.Enable();
    }

    void FixedUpdate()
    {
        // The slider should have a default value of the minimum launch force.
        aimSlider.value = minLaunchForce;

        // If the max force has been exceeded and the shell hasn't yet been launched...
        if (m_CurrentLaunchForce >= maxLaunchForce && !m_Fired)
        {
            // ... use the max force and launch the shell.
            m_CurrentLaunchForce = maxLaunchForce;
            Fire();
        }
        // Otherwise, if the fire button has just started being pressed...
        else if (m_FireButtonPressedThisFrame && !m_FireButtonDown)
        {
            // ... reset the fired flag and reset the launch force.
            // m_Fired = false;
            m_CurrentLaunchForce = minLaunchForce;

            // Change the clip to the charging clip and start it playing.
            // shootingAudio.clip = chargingClip;
            // shootingAudio.Play();

        }
        // Otherwise, if the fire button is being held and the shell hasn't been launched yet...
        else if (m_FireButtonDown && !m_Fired)
        {
            // Increment the launch force and update the slider.
            m_CurrentLaunchForce += m_ChargeSpeed * Time.deltaTime;

            aimSlider.value = m_CurrentLaunchForce;
        }
        // Otherwise, if the fire button is released and the shell hasn't been launched yet...
        else if (m_FireButtonReleasedThisFrame && !m_Fired)
        {
            // ... launch the shell.
            Fire();
        }
    }

    void Fire()
    {
        // Set the fired flag so only Fire is only called once.
        m_Fired = true;

        // Reset the button flags.
        m_FireButtonPressedThisFrame = false;
        m_FireButtonReleasedThisFrame = false;
        m_FireButtonDown = false;

        // Create an instance of the shell and store a reference to it's rigidbody.
        Rigidbody shellInstance =
            Instantiate(shell, fireTransform.position, fireTransform.rotation) as Rigidbody;

        var shellExplosion = shellInstance.GetComponent<ShellExplosion>();
        shellExplosion.SourceTank = m_TankAgent;

        // Set the shell's velocity to the launch force in the fire position's forward direction.
        shellInstance.velocity = m_CurrentLaunchForce * fireTransform.forward;

        // Change the clip to the firing clip and play it.
        // shootingAudio.clip = fireClip;
        // shootingAudio.Play();

        // Reset the launch force.  This is a precaution in case of missing button events.
        m_CurrentLaunchForce = minLaunchForce;

        m_TankAgent.AddReward(-0.001f);
    }

    // The callback from the TanksInputActions Player Input asset that is
    // triggered from the "Fire" action.
    void OnFire(InputValue value)
    {
        // We have setup our button press action to be Press and Release
        // trigger behavior in the Press interaction of the Input Action asset.
        // The isPressed property will be true
        // when OnFire is called during initial button press.
        // It will be false when OnFire is called during button release.

    }

    public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
    {
        InitializeAction();
        return (actions.asset, actions);
    }

    void InitializeAction()
    {
        if (actions == null)
        {
            actions = new TanksInputActions();
            actions.Enable();
        }
    }
}
