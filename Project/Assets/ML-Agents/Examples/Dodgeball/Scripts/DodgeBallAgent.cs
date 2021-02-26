using System;
using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class DodgeBallAgent : Agent
{

    [Header("TEAM")]

    public int teamID;
    private AgentCubeMovement m_CubeMovement;
    public ThrowBall ThrowController;

    [Header("HEALTH")] public AgentHealth AgentHealth;

    [Header("SHIELD")] public ShieldController AgentShield;

    [Header("INPUT")]
    public DodgeBallAgentInput input;

    [Header("INVENTORY")]
    public int currentNumberOfBalls;
    public List<DodgeBall> currentlyHeldBalls;

    public bool UseVectorObs;
    private DodgeBallGameController m_GameController;

    private Vector3 m_StartingPos;
    private Quaternion m_StartingRot;
    private Rigidbody m_AgentRb;

    [Header("HIT EFFECTS")] public ParticleSystem HitByParticles;
    public AudioSource HitSoundAudioSource;
    public Queue<DodgeBall> ActiveBallsQueue = new Queue<DodgeBall>();
    // public List<DodgeBall> ActiveBallsList = new List<DodgeBall>();
    public List<Transform> BallUIList = new List<Transform>();

    public bool AnimateEyes;
    public Transform NormalEyes;
    public Transform HitEyes;
    [Header("SOUNDS")] public AudioClip BallPickupAudioClip;
    public AudioClip TauntVoiceAudioClip;
    public AudioClip HurtVoiceAudioClip;
    public AudioClip BallImpactAudioClip;


    [Header("HIT DAMAGE")] public int NumberOfTimesPlayerCanBeHit = 5;
    public int HitPointsRemaining; //how many more times can we be hit

    public bool m_Initialized;
    [HideInInspector]
    public BehaviorParameters m_BehaviorParameters;


    // [Header("DUMB AI HEURISTIC")] public bool DoNotPerformActions;
    //PLAYER STATE TO OBSERVE

    // Start is called before the first frame update
    public override void Initialize()
    {
        m_CubeMovement = GetComponent<AgentCubeMovement>();
        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();

        //        m_Cam = Camera.main;
        m_AgentRb = GetComponent<Rigidbody>();
        input = GetComponent<DodgeBallAgentInput>();
        m_GameController = FindObjectOfType<DodgeBallGameController>();
        m_StartingPos = transform.position;
        m_StartingRot = transform.rotation;
        m_Initialized = true;
    }



    public override void OnEpisodeBegin()
    {

        if (!m_Initialized)
        {
            Initialize();
        }
        ResetAgent();
    }

    public void ResetAgent()
    {
        transform.position = m_StartingPos;
        // transform.rotation = m_StartingRot;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
        ActiveBallsQueue.Clear();
        HitPointsRemaining = NumberOfTimesPlayerCanBeHit;
        currentNumberOfBalls = 0;
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;
        SetActiveBalls(0);
    }

    void SetActiveBalls(int numOfBalls)
    {
        int i = 0;
        foreach (var item in BallUIList)
        {
            var active = i < numOfBalls;
            BallUIList[i].gameObject.SetActive(active);
            i++;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        if (UseVectorObs)
        {
            // sensor.AddObservation((float)StepCount / (float)MaxStep); //Helps with credit assign?
            sensor.AddObservation((float)currentNumberOfBalls/4); //Held DBs Normalized
            sensor.AddObservation((float)HitPointsRemaining/(float)NumberOfTimesPlayerCanBeHit); //Remaining Hit Points Normalized
        //     //            var localVelocity = transform.InverseTransformDirection(m_AgentRb.velocity);
        //     //            sensor.AddObservation(localVelocity.x);
        //     //            sensor.AddObservation(localVelocity.z);
        //     //            sensor.AddObservation(m_Frozen);
        //     //            sensor.AddObservation(m_ShootInput);
        }
        //        else if (useVectorFrozenFlag)
        //        {
        //            sensor.AddObservation(m_Frozen);
        //        }
    }

    public void MoveAgent(ActionSegment<float> act)
    {
        // print("MoveAgent");

        // if (DoNotPerformActions)
        // {
        //     return;
        // }
        //        if (!m_Frozen)
        //        {
        //            var shootCommand = false;
        //        var forwardAxis = act[0];
        //        var rightAxis = act[1];
        //        var rotateAxis = act[2];
        //        var shootAxis = act[3];
        //        m_Shoot = shootAxis > 0;

        //        m_CubeMovement.RotateBody(rotateAxis, forwardAxis);
        //        m_CubeMovement.RunOnGround(m_AgentRb, m_Cam.transform.TransformDirection(new Vector3(0, 0, forwardAxis)));
        //        m_CubeMovement.Strafe(transform.right * rightAxis);

        // if (AgentHealth.Dead)
        // {
        //     return;
        // }
        m_InputV = act[0];
        m_InputH = act[1];
        m_Rotate = act[2];
        m_ThrowInput = act[3];
        m_DashInput = act[4];

        // print($"{m_InputV} {m_InputH}");
        //HANDLE ROTATION
        m_CubeMovement.Look(m_Rotate);


        //HANDLE XZ MOVEMENT
        Vector3 moveDir = Vector3.zero;
        // moveDir = input.Cam.transform.TransformDirection(new Vector3(m_InputH, 0, m_InputV));
        // moveDir.y = 0;
        moveDir = transform.TransformDirection(new Vector3(m_InputH, 0, m_InputV));
        m_CubeMovement.RunOnGround(moveDir);

        // if (AgentShield && act[6] > 0)
        // {
        //     AgentShield.ActivateShield(true);
        // }
        // else
        // {
        //     AgentShield.ActivateShield(false);
        // }


        // if (m_ShootInput > 0 && currentNumberOfBalls > 0)
        // {
        //     gunController.Shoot();
        //     currentNumberOfBalls--;
        // }

        //HANDLE THROWING
        if (m_ThrowInput > 0)
        // if (input.shootPressed)
        {
            ThrowTheBall();
        }


        // if (act[4] > 0 && m_CubeMovement.groundCheck.isGrounded)
        // {
        //     m_CubeMovement.Jump();
        // }
        //HANDLE DASH MOVEMENT
        if (m_DashInput > 0)
        {
            m_CubeMovement.Dash(moveDir);
        }
        //        }


        // if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        // {
        //     m_AgentRb.velocity *= 0.95f;
        // }
    }

    public void ThrowTheBall()
    {
        if (currentNumberOfBalls > 0)
        {
            var db = ActiveBallsQueue.Peek();
            ThrowController.Throw(db, m_BehaviorParameters.TeamId);
            ActiveBallsQueue.Dequeue();
            currentNumberOfBalls--;
            SetActiveBalls(currentNumberOfBalls);
        }
    }

    //    void OnCollisionEnter(Collision col)
    //    {
    //        if (col.gameObject.CompareTag("projectile"))
    //        {
    //            //IMPLEMENT HEALTH MECHANIC
    //        }
    //    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // print("MoveAgent");

        MoveAgent(actionBuffers.ContinuousActions);
    }

    public float m_InputH;
    private float m_InputV;
    private float m_Rotate;

    // void Update()
    // {
    //     //        m_InputH = Input.GetKeyDown(KeyCode.K) ? 1 : Input.GetKeyDown(KeyCode.J) ? -1 : 0; //inputH
    //     if (Input.GetKeyDown(KeyCode.K))
    //     {
    //         rightStrafe = true;
    //     }
    //
    //     if (Input.GetKeyDown(KeyCode.J))
    //     {
    //         leftStrafe = true;
    //     }
    // }

    //    void FixedUpdate()
    //    {
    //        m_InputV = Input.GetKey(KeyCode.W) ? 1 : Input.GetKey(KeyCode.S) ? -1 : 0; //inputV
    //        //        m_InputH = 0;
    //        //        m_InputH += Input.GetKeyDown(KeyCode.Q) ? -1 : 0;
    //        //        m_InputH += Input.GetKeyDown(KeyCode.E) ? 1 : 0;
    //        //        m_InputH = Input.GetKeyDown(KeyCode.E) ? 1 : Input.GetKeyDown(KeyCode.Q) ? -1 : 0; //inputH
    //        m_Rotate = 0;
    //        m_Rotate += Input.GetKey(KeyCode.A) ? -1 : 0;
    //        m_Rotate += Input.GetKey(KeyCode.D) ? 1 : 0;
    //        //        m_Rotate = Input.GetKey(KeyCode.D) ? 1 : Input.GetKey(KeyCode.A) ? -1 : 0; //rotate
    //        m_ShootInput = Input.GetKey(KeyCode.Space) ? 1 : 0; //shoot
    //    }

    // private Vector2 inputMovement;
    // private Vector2 rotateMovement;
    public float m_ThrowInput;
    public float m_DashInput;

    // public void OnMovement(InputAction.CallbackContext value)
    // {
    //     inputMovement = value.ReadValue<Vector2>();
    // }
    //
    // public void OnRotate(InputAction.CallbackContext value)
    // {
    //     rotateMovement = value.ReadValue<Vector2>();
    // }
    // public void OnShoot(InputAction.CallbackContext value)
    // {
    //     //        m_ShootInput = value.canceled? 0: value.ReadValue<float>();
    //     //            m_ShootInput = 0;
    //     if (value.started)
    //     {
    //         print("started");
    //     }
    //     if (value.performed)
    //     {
    //         print("performed" + Time.frameCount);
    //     }
    //     if (!value.canceled)
    //     {
    //         print("not cancelled" + Time.frameCount + m_ThrowInput);
    //         m_ThrowInput = value.ReadValue<float>();
    //         //        m_ShootInput = value.
    //     }
    //     else
    //     {
    //         m_ThrowInput = 0;
    //         print("cancelled" + Time.frameCount + m_ThrowInput);
    //     }
    // }


    IEnumerator ShowHitFace()
    {
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        float timer = 0;
        NormalEyes.gameObject.SetActive(false);
        HitEyes.gameObject.SetActive(true);
        while (timer < .25f)
        {
            timer += Time.deltaTime;
            yield return wait;
        }
        NormalEyes.gameObject.SetActive(true);
        HitEyes.gameObject.SetActive(false);
    }

    public void PlayHitFX()
    {
            ThrowController.impulseSource.GenerateImpulse();
            // HitSoundAudioSource.Play();
            HitSoundAudioSource.PlayOneShot(BallImpactAudioClip, 1f);
            HitSoundAudioSource.PlayOneShot(HurtVoiceAudioClip, 1f);
            HitByParticles.Play();
            if (AnimateEyes)
            {
                StartCoroutine(ShowHitFace());
            }

    }
    private void OnCollisionEnter(Collision col)
    {
        DodgeBall db = col.gameObject.GetComponent<DodgeBall>();
        if (!db)
        {
            return;
        }


        if (db.inPlay) //HIT BY LIVE BALL
        {
            if (db.TeamToIgnore != -1 && db.TeamToIgnore != m_BehaviorParameters.TeamId) //HIT BY LIVE BALL
            {
                m_GameController.PlayerWasHit(this);
                PlayHitFX();
                // if (HitPointsRemaining == 1)
                // {
                //     //RESET ENV
                //     print($"{gameObject.name} Lost.{gameObject.name} was weak:");
                //     //ASSIGN REWARDS
                //     EndEpisode();
                // }
                // else
                // {
                //     HitPointsRemaining--;
                //     //ASSIGN REWARDS
                //
                // }
                print("HIT BY LIVE BALL");
                // if(HitByParticles.isPlaying)
                db.BallIsInPlay(false);
            }
        }
        else //TRY TO PICK IT UP
        {
            if (currentNumberOfBalls < 4)
            {
                PickUpBall(db);
            }
        }
        // if (col.transform.CompareTag("dodgeBall"))
        // {
        // }
    }

    void PickUpBall(DodgeBall db)
    {
        HitSoundAudioSource.PlayOneShot(BallPickupAudioClip, .1f);
        //update counter
        currentNumberOfBalls++;
        SetActiveBalls(currentNumberOfBalls);

        //add to our inventory
        ActiveBallsQueue.Enqueue(db);
        db.BallIsInPlay(true);
        db.gameObject.SetActive(false);
        // ActiveBallsList.Add(db);
    }


    // void Update()
    // {
    //
    // }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // print("Heuristic");

        var contActionsOut = actionsOut.ContinuousActions;

        contActionsOut[0] = input.moveInput.y;
        contActionsOut[1] = input.moveInput.x;
        //        contActionsOut[2] = input.rotateInput.x; //rotate
        contActionsOut[2] = input.rotateInput.x; //rotate
        // print(input.rotateInput.x);
        // contActionsOut[3] = input.shootInput ? 1 : 0; //shoot
        contActionsOut[3] = input.throwPressed ? 1 : 0; //shoot
        // contActionsOut[3] = input.CheckIfInputSinceLastFrame(ref input.shootInput) ? 1 : 0; //jump
        // contActionsOut[3] = input.CheckIfInputSinceLastFrame(ref input.shootPressed) ? 1 : 0; //throw
        contActionsOut[4] = input.CheckIfInputSinceLastFrame(ref input.dashInput) ? 1 : 0; //dash
        // contActionsOut[5] = input.CheckIfInputSinceLastFrame(ref input.jumpInput) ? 1 : 0; //jump
        //                                                                                    //        contActionsOut[4] = input.jumpInput ? 1 : 0; //jump
        //                                                                                    //        contActionsOut[5] = input.dashInput ? 1 : 0; //dash
        // contActionsOut[6] = input.shieldInput ? 1 : 0; //shield
        // if (input.jumpInput)
        // {
        //     print($"Agent: Jump: {input.jumpInput} : {Time.frameCount}");
        //
        // }

        //        contActionsOut[0] = inputMovement.y;
        //        contActionsOut[1] = inputMovement.x;
        //        contActionsOut[2] = rotateMovement.x;


        //        m_InputH = 0;
        //        if (leftStrafe)
        //        {
        //            //            print("leftstrafe");
        //            m_InputH += -1;
        //            leftStrafe = false;
        //        }
        //
        //        if (rightStrafe)
        //        {
        //            //            print("rightstrafe");
        //            m_InputH += 1;
        //            rightStrafe = false;
        //        }
        //
        //        contActionsOut[1] = m_InputH; //inputH
    }
}
