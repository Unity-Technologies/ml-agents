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

    public float m_InputH;
    private float m_InputV;
    private float m_Rotate;
    public float m_ThrowInput;
    public float m_DashInput;

    public override void Initialize()
    {
        m_CubeMovement = GetComponent<AgentCubeMovement>();
        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();

        //        m_Cam = Camera.main;
        m_AgentRb = GetComponent<Rigidbody>();
        input = GetComponent<DodgeBallAgentInput>();
        m_GameController = GetComponentInParent<DodgeBallGameController>();
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
            sensor.AddObservation(ThrowController.coolDownWait); //Held DBs Normalized
            sensor.AddObservation((float)currentNumberOfBalls/4); //Held DBs Normalized
            sensor.AddObservation((float)HitPointsRemaining/(float)NumberOfTimesPlayerCanBeHit); //Remaining Hit Points Normalized
            sensor.AddObservation(Vector3.Dot(m_AgentRb.velocity, m_AgentRb.transform.forward));
            sensor.AddObservation(Vector3.Dot(m_AgentRb.velocity, m_AgentRb.transform.right));
            // sensor.AddObservation(Vector3.Dot(m_AgentRb.angularVelocity, m_AgentRb.transform.forward));
            // sensor.AddObservation(Vector3.Dot(m_AgentRb.velocity, m_AgentRb.transform.right));
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

    // public void MoveAgent(ActionSegment<float> act)
    public void MoveAgent(ActionBuffers actionBuffers)
    {
        // if (DoNotPerformActions)
        // {
        //     return;
        // }
        var continuousActions = actionBuffers.ContinuousActions;
        var discreteActions = actionBuffers.DiscreteActions;

        m_InputV = continuousActions[0];
        m_InputH = continuousActions[1];
        m_Rotate = continuousActions[2];
        m_ThrowInput = (int)discreteActions[0];;
        m_DashInput = (int)discreteActions[1];;

        //HANDLE ROTATION
        m_CubeMovement.Look(m_Rotate);

        //HANDLE XZ MOVEMENT
        var moveDir = transform.TransformDirection(new Vector3(m_InputH, 0, m_InputV));
        m_CubeMovement.RunOnGround(moveDir);

        // if (AgentShield && act[6] > 0)
        // {
        //     AgentShield.ActivateShield(true);
        // }
        // else
        // {
        //     AgentShield.ActivateShield(false);
        // }

        //HANDLE THROWING
        if (m_ThrowInput > 0)
        // if (input.shootPressed)
        {
            ThrowTheBall();
        }

        //HANDLE DASH MOVEMENT
        if (m_DashInput > 0)
        {
            m_CubeMovement.Dash(moveDir);
        }
    }

    public void ThrowTheBall()
    {
        if (currentNumberOfBalls > 0 && !ThrowController.coolDownWait)
        {
            var db = ActiveBallsQueue.Peek();
            ThrowController.Throw(db, m_BehaviorParameters.TeamId);
            ActiveBallsQueue.Dequeue();
            currentNumberOfBalls--;
            SetActiveBalls(currentNumberOfBalls);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // print("MoveAgent");

        MoveAgent(actionBuffers);
    }

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
                print("HIT BY LIVE BALL");
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

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var contActionsOut = actionsOut.ContinuousActions;

        contActionsOut[0] = input.moveInput.y;
        contActionsOut[1] = input.moveInput.x;
        contActionsOut[2] = input.rotateInput.x; //rotate
        // contActionsOut[3] = input.throwPressed ? 1 : 0; //shoot
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = input.CheckIfInputSinceLastFrame(ref input.throwPressed) ? 1 : 0; //dash
        // contActionsOut[3] = input.throwInput; //shoot
        discreteActionsOut[1] = input.CheckIfInputSinceLastFrame(ref input.dashInput) ? 1 : 0; //dash
    }
}
