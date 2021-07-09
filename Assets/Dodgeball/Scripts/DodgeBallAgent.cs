using System;
using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
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
    public Transform HomeBaseLocation;
    public Transform TeamFlag;
    private DodgeBallGameController m_GameController;

    private Vector3 m_StartingPos;
    private Quaternion m_StartingRot;
    [HideInInspector]
    public Rigidbody AgentRb;

    [Header("HIT EFFECTS")] public ParticleSystem HitByParticles;
    private AudioSource m_BallImpactAudioSource;
    private AudioSource m_StunnedAudioSource;
    public Queue<DodgeBall> ActiveBallsQueue = new Queue<DodgeBall>();
    public List<Transform> BallUIList = new List<Transform>();

    public bool AnimateEyes;
    public Transform NormalEyes;
    public Transform HitEyes;
    public Transform StunnedEyes;
    public ParticleSystem StunnedEffect;
    public Transform Flag;
    public Transform StunnedCollider;
    
    [Header("ANIMATIONS")] public Animator FlagAnimator;
    public Animator VictoryDanceAnimation;

    [HideInInspector]
    public int NumberOfTimesPlayerCanBeHit = 5;
    [HideInInspector]
    public int HitPointsRemaining; //how many more times can we be hit

    [Header("OTHER")] public bool m_PlayerInitialized;
    [HideInInspector]
    public BehaviorParameters m_BehaviorParameters;

    public float m_InputH;
    private Vector3 m_HomeBasePosition;
    private Vector3 m_HomeDirection;
    private float m_InputV;
    private float m_Rotate;
    public float m_ThrowInput;
    public float m_DashInput;
    private bool m_FirstInitialize = true;
    private bool m_DashCoolDownReady;
    private bool m_IsStunned;
    public float m_StunTime;
    private float m_OpponentHasFlagPenalty;
    private float m_TeamHasFlagBonus;
    private float m_BallHoldBonus = 0.0f;
    private float m_LocationNormalizationFactor = 80.0f; // About the size of a reasonable stage
    private EnvironmentParameters m_EnvParameters;

    public BufferSensorComponent m_BallBuffer;
    public BufferSensorComponent m_OtherAgentsBuffer;
    float[] ballOneHot = new float[5];

    //is the current step a decision step for the agent
    private bool m_IsDecisionStep;

    [HideInInspector]
    //because heuristic only runs every 5 fixed update steps, the input for a human feels really bad
    //set this to true on an agent that you want to be human playable and it will collect input every
    //FixedUpdate tick instead of ever decision step
    public bool disableInputCollectionInHeuristicCallback;

    public override void Initialize()
    {

        //SETUP STUNNED AS
        m_StunnedAudioSource = gameObject.AddComponent<AudioSource>();
        m_StunnedAudioSource.spatialBlend = 1;
        m_StunnedAudioSource.maxDistance = 250;

        //SETUP IMPACT AS
        m_BallImpactAudioSource = gameObject.AddComponent<AudioSource>();
        m_BallImpactAudioSource.spatialBlend = 1;
        m_BallImpactAudioSource.maxDistance = 250;

        var bufferSensors = GetComponentsInChildren<BufferSensorComponent>();
        m_OtherAgentsBuffer = bufferSensors[0];

        m_CubeMovement = GetComponent<AgentCubeMovement>();
        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();

        AgentRb = GetComponent<Rigidbody>();
        input = GetComponent<DodgeBallAgentInput>();
        m_GameController = GetComponentInParent<DodgeBallGameController>();

        //Make sure ThrowController is set up to play sounds
        ThrowController.PlaySound = m_GameController.ShouldPlayEffects;

        if (m_FirstInitialize)
        {
            m_StartingPos = transform.position;
            m_StartingRot = transform.rotation;
            //If we don't have a home base, just use the starting position.
            if (HomeBaseLocation is null)
            {
                m_HomeBasePosition = m_StartingPos;
                m_HomeDirection = transform.forward;
            }
            else
            {
                m_HomeBasePosition = HomeBaseLocation.position;
                m_HomeDirection = HomeBaseLocation.forward;
            }
            m_FirstInitialize = false;
            Flag.gameObject.SetActive(false);
        }
        m_EnvParameters = Academy.Instance.EnvironmentParameters;
        GetAllParameters();
    }

    //Get all environment parameters for agent
    private void GetAllParameters()
    {
        m_StunTime = m_EnvParameters.GetWithDefault("stun_time", 10.0f);
        m_OpponentHasFlagPenalty = m_EnvParameters.GetWithDefault("opponent_has_flag_penalty", 0f);
        m_TeamHasFlagBonus = m_EnvParameters.GetWithDefault("team_has_flag_bonus", 0f);
        m_BallHoldBonus = m_EnvParameters.GetWithDefault("ball_hold_bonus", 0f);
    }

    public void ResetAgent()
    {
        GetAllParameters();
        StopAllCoroutines();
        transform.position = m_StartingPos;
        AgentRb.constraints = RigidbodyConstraints.FreezeRotation;
        if (m_GameController.CurrentSceneType == DodgeBallGameController.SceneType.Game || m_GameController.CurrentSceneType == DodgeBallGameController.SceneType.Movie)
        {
            transform.rotation = m_StartingRot;
        }
        else //Training Mode so we want random rotations
        {
            transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
        }
        ActiveBallsQueue.Clear();
        currentNumberOfBalls = 0;
        AgentRb.velocity = Vector3.zero;
        AgentRb.angularVelocity = Vector3.zero;
        SetActiveBalls(0);
        NormalEyes.gameObject.SetActive(true);
        HitEyes.gameObject.SetActive(false);
        HasEnemyFlag = false;
        Stunned = false;
        AgentRb.drag = 4;
        AgentRb.angularDrag = 1;
        Dancing = false;
    }

    //Set agent to stunned, then send it back to spawn point.
    public void StunAndReset()
    {
        DropAllBalls();
        StartCoroutine(StunThenReset());
    }

    IEnumerator StunThenReset()
    {
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        float timer = 0;
        Stunned = true;
        while (timer < m_StunTime)
        {
            timer += Time.fixedDeltaTime;
            yield return wait;
        }
        //Play poof as agent gets removed from level
        if (m_GameController.usePoofParticlesOnElimination)
        {
            m_GameController.PlayParticleAtPosition(transform.position);
        }
        ResetAgent();
        //Play second poof as agent respawns
        if (m_GameController.usePoofParticlesOnElimination)
        {
            m_GameController.PlayParticleAtPosition(transform.position);
        }
    }

    //Set the number of active balls.
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

    private int m_AgentStepCount; //current agent step
    void FixedUpdate()
    {
        m_DashCoolDownReady = m_CubeMovement.dashCoolDownTimer > m_CubeMovement.dashCoolDownDuration;
        if (StepCount % 5 == 0)
        {
            m_IsDecisionStep = true;
            m_AgentStepCount++;
        }
        // Handle if flag gets home
        if (Vector3.Distance(m_HomeBasePosition, transform.position) <= 3.0f && HasEnemyFlag)
        {
            m_GameController.FlagWasBroughtHome(this);
        }
    }

    //Collect observations, to be used by the agent in ML-Agents.
    public override void CollectObservations(VectorSensor sensor)
    {
        AddReward(m_BallHoldBonus * (float)currentNumberOfBalls);
        if (UseVectorObs)
        {
            sensor.AddObservation(ThrowController.coolDownWait); //Held DBs Normalized
            sensor.AddObservation(Stunned);
            Array.Clear(ballOneHot, 0, 5);
            ballOneHot[currentNumberOfBalls] = 1f;
            sensor.AddObservation(ballOneHot); //Held DBs Normalized
            sensor.AddObservation((float)HitPointsRemaining / (float)NumberOfTimesPlayerCanBeHit); //Remaining Hit Points Normalized

            sensor.AddObservation(Vector3.Dot(AgentRb.velocity, AgentRb.transform.forward));
            sensor.AddObservation(Vector3.Dot(AgentRb.velocity, AgentRb.transform.right));
            sensor.AddObservation(transform.InverseTransformDirection(m_HomeDirection));
            sensor.AddObservation(m_DashCoolDownReady);  // Remaining cooldown, capped at 1
            // Location to base
            sensor.AddObservation(GetRelativeCoordinates(m_HomeBasePosition));

            sensor.AddObservation(HasEnemyFlag);
        }

        List<DodgeBallGameController.PlayerInfo> teamList;
        List<DodgeBallGameController.PlayerInfo> opponentsList;
        if (m_BehaviorParameters.TeamId == 0)
        {
            teamList = m_GameController.Team0Players;
            opponentsList = m_GameController.Team1Players;
        }
        else
        {
            teamList = m_GameController.Team1Players;
            opponentsList = m_GameController.Team0Players;
        }

        foreach (var info in teamList)
        {
            if (info.Agent != this && info.Agent.gameObject.activeInHierarchy)
            {
                m_OtherAgentsBuffer.AppendObservation(GetOtherAgentData(info));
            }
            if (info.Agent.HasEnemyFlag) // If anyone on my team has the enemy flag
            {
                AddReward(m_TeamHasFlagBonus);
            }
        }
        //Only opponents who picked up the flag are visible
        var currentFlagPosition = TeamFlag.transform.position;
        int numEnemiesRemaining = 0;
        bool enemyHasFlag = false;
        foreach (var info in opponentsList)
        {
            if (info.Agent.gameObject.activeInHierarchy)
            {
                numEnemiesRemaining++;
            }
            if (info.Agent.HasEnemyFlag)
            {
                enemyHasFlag = true;
                currentFlagPosition = info.Agent.transform.position;
                AddReward(m_OpponentHasFlagPenalty); // If anyone on the opposing team has a flag
            }
        }
        var portionOfEnemiesRemaining = (float)numEnemiesRemaining / (float)opponentsList.Count;

        //Different observation for different mode. Enemy Has Flag is only relevant to CTF
        if (m_GameController.GameMode == DodgeBallGameController.GameModeType.CaptureTheFlag)
        {
            sensor.AddObservation(enemyHasFlag);
        }
        else
        {
            sensor.AddObservation(numEnemiesRemaining);
        }

        //Location to flag
        sensor.AddObservation(GetRelativeCoordinates(currentFlagPosition));
    }

    //Get normalized position relative to agent's current position.
    private float[] GetRelativeCoordinates(Vector3 pos)
    {
        Vector3 relativeHome = transform.InverseTransformPoint(pos);
        var relativeCoordinate = new float[2];
        relativeCoordinate[0] = (relativeHome.x) / m_LocationNormalizationFactor;
        relativeCoordinate[1] = (relativeHome.z) / m_LocationNormalizationFactor;
        return relativeCoordinate;
    }

    //Get information of teammate
    private float[] GetOtherAgentData(DodgeBallGameController.PlayerInfo info)
    {
        var otherAgentdata = new float[8];
        otherAgentdata[0] = (float)info.Agent.HitPointsRemaining / (float)NumberOfTimesPlayerCanBeHit;
        var relativePosition = transform.InverseTransformPoint(info.Agent.transform.position);
        otherAgentdata[1] = relativePosition.x / m_LocationNormalizationFactor;
        otherAgentdata[2] = relativePosition.z / m_LocationNormalizationFactor;
        otherAgentdata[3] = info.TeamID == teamID ? 0.0f : 1.0f;
        otherAgentdata[4] = info.Agent.HasEnemyFlag ? 1.0f : 0.0f;
        otherAgentdata[5] = info.Agent.Stunned ? 1.0f : 0.0f;
        var relativeVelocity = transform.InverseTransformDirection(info.Agent.AgentRb.velocity);
        otherAgentdata[6] = relativeVelocity.x / 30.0f;
        otherAgentdata[7] = relativeVelocity.z / 30.0f;
        return otherAgentdata;

    }

    //Excute agent movement
    public void MoveAgent(ActionBuffers actionBuffers)
    {
        if (Stunned)
        {
            return;
        }
        var continuousActions = actionBuffers.ContinuousActions;
        var discreteActions = actionBuffers.DiscreteActions;

        m_InputV = continuousActions[0];
        m_InputH = continuousActions[1];
        m_Rotate = continuousActions[2];
        m_ThrowInput = (int)discreteActions[0];
        m_DashInput = (int)discreteActions[1];

        //HANDLE ROTATION
        m_CubeMovement.Look(m_Rotate);

        //HANDLE XZ MOVEMENT
        var moveDir = transform.TransformDirection(new Vector3(m_InputH, 0, m_InputV));
        m_CubeMovement.RunOnGround(moveDir);

        //perform discrete actions only once between decisions
        if (m_IsDecisionStep)
        {
            m_IsDecisionStep = false;
            //HANDLE THROWING
            if (m_ThrowInput > 0)
            {
                ThrowTheBall();
            }
            //HANDLE DASH MOVEMENT
            if (m_DashInput > 0 && m_DashCoolDownReady)
            {
                m_CubeMovement.Dash(moveDir);
            }
        }
    }

    public void ThrowTheBall()
    {
        if (currentNumberOfBalls > 0 && !ThrowController.coolDownWait)
        {
            var db = ActiveBallsQueue.Peek();
            ThrowController.Throw(db, this, m_BehaviorParameters.TeamId);
            ActiveBallsQueue.Dequeue();
            currentNumberOfBalls--;
            SetActiveBalls(currentNumberOfBalls);
        }
    }

    public void DropAllBalls()
    {
        while (currentNumberOfBalls > 0)
        {
            var db = ActiveBallsQueue.Peek();
            ThrowController.Drop(db);
            ActiveBallsQueue.Dequeue();
            currentNumberOfBalls--;
            SetActiveBalls(currentNumberOfBalls);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
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
        // Don't reshow face if this hit resulted in a stun
        if (!Stunned)
        {
            NormalEyes.gameObject.SetActive(true);
            HitEyes.gameObject.SetActive(false);
        }
    }

    public void PlayHitFX()
    {
        // Only shake if player object
        if (ThrowController.UseScreenShake && m_GameController.PlayerGameObject == gameObject)
        {
            ThrowController.impulseSource.GenerateImpulse();
        }
        PlayBallThwackSound();
        HitByParticles.Play();
        if (AnimateEyes)
        {
            StartCoroutine(ShowHitFace());
        }
    }

    public void PlayBallThwackSound()
    {
        if (m_GameController.ShouldPlayEffects)
        {
            m_BallImpactAudioSource.pitch = Random.Range(2f, 3f);
            m_BallImpactAudioSource.PlayOneShot(m_GameController.BallImpactClip2, 1f);
            m_BallImpactAudioSource.PlayOneShot(m_GameController.BallImpactClip1, 1f);
        }
    }
    
    public void PlayStunnedVoice()
    {
        if (m_GameController.ShouldPlayEffects)
        {
            m_StunnedAudioSource.pitch = Random.Range(.3f, .8f);
            m_StunnedAudioSource.PlayOneShot(m_GameController.HurtVoiceAudioClip, 1f);
        }
    }

    void ResetHeldFlag()
    {
        // Reset flag rotation
        Transform mesh = Flag.gameObject.transform.GetChild(0);
        mesh.localRotation = Quaternion.identity;
    }

    public bool HasEnemyFlag
    {
        get => Flag.gameObject.activeSelf;
        set
        {
            if (value)
            {
                Flag.gameObject.SetActive(true);
                ResetHeldFlag();
            }
            else
            {
                ResetHeldFlag();
                Flag.gameObject.SetActive(false);
            }
        }
    }

    public bool Stunned
    {
        get => m_IsStunned;
        set
        {
            if (value)
            {
                NormalEyes.gameObject.SetActive(false);
                HitEyes.gameObject.SetActive(false);
                StunnedEyes.gameObject.SetActive(true);
                StunnedEffect.Play();
                StunnedCollider.gameObject.SetActive(true);
                if (m_GameController.GameMode == DodgeBallGameController.GameModeType.CaptureTheFlag)
                {
                    AgentRb.mass = 100000;
                }
            }
            else
            {
                NormalEyes.gameObject.SetActive(true);
                HitEyes.gameObject.SetActive(false);
                StunnedEyes.gameObject.SetActive(false);
                StunnedEffect.Stop();
                StunnedCollider.gameObject.SetActive(false);
                AgentRb.mass = 10;
            }
            m_IsStunned = value;
        }
    }

    public bool Dancing
    {
        set
        {
            if (value)
            {
                VictoryDanceAnimation.enabled = true;
                m_IsStunned = true;
                Flag.gameObject.SetActive(false);
            }
            else
            {
                VictoryDanceAnimation.enabled = false;
                m_IsStunned = false;
                Flag.gameObject.SetActive(HasEnemyFlag);
            }
        }
    }

    private void OnCollisionEnter(Collision col)
    {
        // Ignore all collisions when stunned
        if (Stunned)
        {
            return;
        }
        DodgeBall db = col.gameObject.GetComponent<DodgeBall>();
        if (!db)
        {
            if (m_GameController.GameMode == DodgeBallGameController.GameModeType.CaptureTheFlag)
            {
                // Check if it is a flag
                if (col.gameObject.tag == "purpleFlag" && teamID == 0 || col.gameObject.tag == "blueFlag" && teamID == 1)
                {
                    m_GameController.FlagWasTaken(this);
                }
                else if (col.gameObject.tag == "purpleFlag" && teamID == 1 || col.gameObject.tag == "blueFlag" && teamID == 0)
                {
                    m_GameController.ReturnFlag(this);
                }
                DodgeBallAgent hitAgent = col.gameObject.GetComponent<DodgeBallAgent>();
                if (hitAgent && HasEnemyFlag && m_GameController.FlagCarrierKnockback)
                {
                    if (hitAgent.teamID != teamID && !hitAgent.Stunned)
                    {
                        if (m_GameController.ShouldPlayEffects)
                        {
                            m_BallImpactAudioSource.PlayOneShot(m_GameController.FlagHitClip, 1f);
                        }
                        // Play Flag Whack
                        if (FlagAnimator != null)
                        {
                            FlagAnimator.SetTrigger("FlagSwing");
                        }
                        hitAgent.PlayHitFX();
                        var moveDirection = hitAgent.transform.position - transform.position;
                        hitAgent.AgentRb.AddForce(moveDirection * 150, ForceMode.Impulse);
                    }
                }
            }
            return;
        }

        if (db.inPlay) //HIT BY LIVE BALL
        {
            if (db.TeamToIgnore != -1 && db.TeamToIgnore != m_BehaviorParameters.TeamId) //HIT BY LIVE BALL
            {
                PlayHitFX();
                m_GameController.PlayerWasHit(this, db.thrownBy);
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
        if (m_GameController.ShouldPlayEffects)
        {
            m_BallImpactAudioSource.PlayOneShot(m_GameController.BallPickupClip, .1f);
        }
        //update counter
        currentNumberOfBalls++;
        SetActiveBalls(currentNumberOfBalls);

        //add to our inventory
        ActiveBallsQueue.Enqueue(db);
        db.BallIsInPlay(true);
        db.gameObject.SetActive(false);
    }

    //Used for human input
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (disableInputCollectionInHeuristicCallback || m_IsStunned)
        {
            return;
        }
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = input.moveInput.y;
        contActionsOut[1] = input.moveInput.x;
        contActionsOut[2] = input.rotateInput * 3; //rotate
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = input.CheckIfInputSinceLastFrame(ref input.m_throwPressed) ? 1 : 0; //dash
        discreteActionsOut[1] = input.CheckIfInputSinceLastFrame(ref input.m_dashPressed) ? 1 : 0; //dash
    }
}
