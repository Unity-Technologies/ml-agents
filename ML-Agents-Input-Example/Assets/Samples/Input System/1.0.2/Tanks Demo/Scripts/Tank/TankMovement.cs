using Unity.MLAgents.Extensions.Input;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;

public class TankMovement : MonoBehaviour
{
    [FormerlySerializedAs("m_PlayerNumber")]
    public int playerNumber = 1;              // Used to identify which tank belongs to which player.  This is set by this tank's manager.
    [FormerlySerializedAs("m_Speed")]
    public float speed = 12f;                 // How fast the tank moves forward and back.
    [FormerlySerializedAs("m_TurnSpeed")]
    public float turnSpeed = 180f;            // How fast the tank turns in degrees per second.
    [FormerlySerializedAs("m_MovementAudio")]
    public AudioSource movementAudio;         // Reference to the audio source used to play engine sounds. NB: different to the shooting audio source.
    [FormerlySerializedAs("m_EngineIdling")]
    public AudioClip engineIdling;            // Audio to play when the tank isn't moving.
    [FormerlySerializedAs("m_EngineDriving")]
    public AudioClip engineDriving;           // Audio to play when the tank is moving.
    [FormerlySerializedAs("m_PitchRange")]
    public float pitchRange = 0.2f;           // The amount by which the pitch of the engine noises can vary.

    Rigidbody m_Rigidbody;              // Reference used to move the tank.
    float m_MovementInputValue;         // The current value of the movement input.
    float m_TurnInputValue;             // The current value of the turn input.
    float m_OriginalPitch;              // The pitch of the audio source at the start of the scene.
    ParticleSystem[] m_ParticleSystems; // References to all the particles systems used by the Tanks
    public TanksInputActions actions;

    InputAction m_Gas;
    InputAction m_Brake;
    InputAction m_Turret;

    void Awake()
    {


        m_Rigidbody = GetComponent<Rigidbody>();
    }

    void OnEnable()
    {
        // When the tank is turned on, make sure it's not kinematic.
        m_Rigidbody.isKinematic = false;

        // Also reset the input values.
        m_MovementInputValue = 0f;
        m_TurnInputValue = 0f;

        // We grab all the Particle systems child of that Tank to be able to Stop/Play them on Deactivate/Activate
        // It is needed because we move the Tank when spawning it, and if the Particle System is playing while we do that
        // it "think" it move from (0,0,0) to the spawn point, creating a huge trail of smoke
        m_ParticleSystems = GetComponentsInChildren<ParticleSystem>();
        for (int i = 0; i < m_ParticleSystems.Length; ++i)
        {
            m_ParticleSystems[i].Play();
        }
        var (_, a) = GetComponent<IInputActionAssetProvider>().GetInputActionAsset();
        actions = a as TanksInputActions;
        Assert.IsNotNull(actions);

        actions.Player.Gas.started += GasOnperformed;
        actions.Player.Gas.performed += GasOnperformed;
        actions.Player.Gas.canceled += GasOnperformed;
        actions.Player.Brake.performed += BrakeOnperformed;
        actions.Player.Brake.canceled += BrakeOnperformed;
        actions.Player.Brake.started += BrakeOnperformed;
        actions.Player.Turret.performed += TurretOnperformed;
        actions.Player.Turret.canceled += TurretOnperformed;
        actions.Player.Turret.started += TurretOnperformed;
    }

    void OnDisable()
    {
        // When the tank is turned off, set it to kinematic so it stops moving.
        m_Rigidbody.isKinematic = true;

        // Stop all particle system so it "reset" it's position to the actual one instead of thinking we moved when spawning
        for (int i = 0; i < m_ParticleSystems.Length; ++i)
        {
            m_ParticleSystems[i].Stop();
        }

        actions.Player.Gas.started -= GasOnperformed;
        actions.Player.Gas.started -= GasOnperformed;
        actions.Player.Gas.started -= GasOnperformed;
        actions.Player.Brake.started -= BrakeOnperformed;
        actions.Player.Brake.started -= BrakeOnperformed;
        actions.Player.Brake.started -= BrakeOnperformed;
        actions.Player.Turret.started -= TurretOnperformed;
        actions.Player.Turret.started -= TurretOnperformed;
        actions.Player.Turret.started -= TurretOnperformed;
    }

    void Start()
    {
        // Store the original pitch of the audio source.
        m_OriginalPitch = movementAudio.pitch;
        actions.Enable();
    }

    void Update()
    {
        // EngineAudio();
    }

    void EngineAudio()
    {
        // If there is no input (the tank is stationary)...
        if (Mathf.Abs(m_MovementInputValue) < 0.1f && Mathf.Abs(m_TurnInputValue) < 0.1f)
        {
            // ... and if the audio source is currently playing the driving clip...
            if (movementAudio.clip == engineDriving)
            {
                // ... change the clip to idling and play it.
                movementAudio.clip = engineIdling;
                movementAudio.pitch = Random.Range(m_OriginalPitch - pitchRange, m_OriginalPitch + pitchRange);
                movementAudio.Play();
            }
        }
        else
        {
            // Otherwise if the tank is moving and if the idling clip is currently playing...
            if (movementAudio.clip == engineIdling)
            {
                // ... change the clip to driving and play.
                movementAudio.clip = engineDriving;
                movementAudio.pitch = Random.Range(m_OriginalPitch - pitchRange, m_OriginalPitch + pitchRange);
                movementAudio.Play();
            }
        }
    }

    void FixedUpdate()
    {
        // Adjust the rigidbodies position and orientation in FixedUpdate.
        var actionsPlayer = actions.Player;
        // TurretOnperformed(actionsPlayer.Turret);
        // GasOnperformed(actionsPlayer.Gas);
        // BrakeOnperformed(actionsPlayer.Brake);
        Move();
        Turn();
    }

    void Move()
    {
        // Create a vector in the direction the tank is facing with a magnitude based on the input, speed and the time between frames.
        Vector3 movement = transform.forward * m_MovementInputValue * speed * Time.deltaTime;

        // Apply this movement to the rigidbody's position.
        m_Rigidbody.MovePosition(m_Rigidbody.position + movement);
    }

    void TurretOnperformed(InputAction.CallbackContext obj)
    {
        m_TurnInputValue = obj.ReadValue<float>();
    }

    void GasOnperformed(InputAction.CallbackContext obj)
    {
        m_MovementInputValue = obj.ReadValue<float>();
    }

    void BrakeOnperformed(InputAction.CallbackContext obj)
    {
        m_MovementInputValue = -obj.ReadValue<float>();
    }

    void Turn()
    {
        // Determine the number of degrees to be turned based on the input, speed and time between frames.
        float turn = m_TurnInputValue * turnSpeed * Time.deltaTime;

        // Make this into a rotation in the y axis.
        Quaternion turnRotation = Quaternion.Euler(0f, turn, 0f);

        // Apply this rotation to the rigidbody's rotation.
        m_Rigidbody.MoveRotation(m_Rigidbody.rotation * turnRotation);
    }

    // The callback from the TanksInputActions Player Input asset that is
    // triggered from the "Gas" action.
    void OnGas(InputValue value)
    {
        m_MovementInputValue = value.Get<float>();
    }

    // The callback from the TanksInputActions Player Input asset that is
    // triggered from the "Brake" action.
    void OnBrake(InputValue value)
    {
        m_MovementInputValue = value.Get<float>() * -1.0f;
    }

    // The callback from the TanksInputActions Player Input asset that is
    // triggered from the "Turret" action.
    void OnTurret(InputValue value)
    {
        m_TurnInputValue = value.Get<float>();
    }

    // The callback from the TanksInputActions Player Input asset that is
    // triggered from the "Pause" action.
    void OnPause(InputValue value)
    {
        SceneManager.LoadScene("ButtonRemapScreen");
    }
}
