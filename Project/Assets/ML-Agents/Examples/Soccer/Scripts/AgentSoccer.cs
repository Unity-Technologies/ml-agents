using UnityEngine;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

public enum Team
{
    Blue = 0,
    Purple = 1
}

public class AgentSoccer : Agent
{
    // Note that that the detectable tags are different for the blue and purple teams. The order is
    // * ball
    // * own goal
    // * opposing goal
    // * wall
    // * own teammate
    // * opposing player

    public SoccerEnvController GameController;
    public enum Position
    {
        Striker,
        Goalie,
        Generic
    }

    [HideInInspector]
    public Team team;
    float m_KickPower;
    // The coefficient for the reward for colliding with a ball. Set using curriculum.
    float m_BallTouch;
    public Position position;

    const float k_Power = 3000f;
    float m_Existential;
    float m_LateralSpeed;
    float m_ForwardSpeed;
    float m_RotateSpeed = 1.5f;


    [HideInInspector]
    public Rigidbody agentRb;
    SoccerSettings m_SoccerSettings;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public float rotSign;

    EnvironmentParameters m_ResetParams;


    [Header("ACTUATED RAY SENSOR")]
    //Actuated RayS
    public bool UseVectorObs = true;
    public bool UseActuatedRaycastSensor = false;
    // public List<RayPerceptionSensorComponent3D> RaySensorsList = new List<RayPerceptionSensorComponent3D>();
    public RayPerceptionSensorComponent3D RaySensor;

    public Vector2 MinMaxRayAngles = new Vector2(25, 120);
    public Vector2 MinMaxSpherecastRadius = new Vector2(.25f, 1);
    public float CurrentRayAngleLerp = .5f;
    public float CurrentSpherecastRadiusLerp = .5f;

    public override void Initialize()
    {
        m_Existential = 1f / MaxStep;
        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        if (position == Position.Goalie)
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.0f;
        }
        else if (position == Position.Striker)
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.3f;
        }
        else
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.5f;
        }
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        if (UseVectorObs)
        {
            // // sensor.AddObservation((float)StepCount / (float)MaxStep); //Helps with credit assign?
            // sensor.AddObservation(ThrowController.coolDownWait); //Held DBs Normalized
            // sensor.AddObservation((float)currentNumberOfBalls/4); //Held DBs Normalized
            // // sensor.AddObservation((float)HitPointsRemaining/(float)NumberOfTimesPlayerCanBeHit); //Remaining Hit Points Normalized
            // sensor.AddObservation((float)HitPointsRemaining/(float)m_GameController.PlayerMaxHitPoints); //Remaining Hit Points Normalized
            // sensor.AddObservation(Vector3.Dot(m_AgentRb.velocity, m_AgentRb.transform.forward)); //forward speed
            // sensor.AddObservation(Vector3.Dot(m_AgentRb.velocity, m_AgentRb.transform.right)); //lateral speed
            // sensor.AddObservation(m_AgentRb.angularVelocity);

            if (UseActuatedRaycastSensor)
            {
                sensor.AddObservation(CurrentRayAngleLerp);
                sensor.AddObservation(CurrentSpherecastRadiusLerp);
                // sensor.AddObservation(GameController.LocalBallVelocity);
                // sensor.AddObservation(GameController.LocalBallPositionNormalizedToFieldSize);
            }
        }
    }

    public void UpdateSensors()
    {

    }

    public void MoveAgent(ActionSegment<float> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        m_KickPower = 0f;
        var forward = Mathf.Clamp(act[0], -1f, 1f);
        var right = Mathf.Clamp(act[1], -1f, 1f);
        var rotate = Mathf.Clamp(act[2], -1f, 1f);

        if (forward > 0)
        {
            m_KickPower = forward;
        }

        dirToGo = transform.forward * forward * m_ForwardSpeed;
        dirToGo += transform.right * right * m_LateralSpeed;
        rotateDir = -transform.up * rotate * m_RotateSpeed;


        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed,
            ForceMode.VelocityChange);


        // ACTUATED SENSOR STUFF
        CurrentRayAngleLerp = (act[3] + 1)/2;
        CurrentSpherecastRadiusLerp = (act[4] + 1)/2;
        if (UseActuatedRaycastSensor)
        {
                RaySensor.MaxRayDegrees = Mathf.Lerp(MinMaxRayAngles.x, MinMaxRayAngles.y, CurrentRayAngleLerp);
                RaySensor.SphereCastRadius = Mathf.Lerp(MinMaxSpherecastRadius.x, MinMaxSpherecastRadius.y, CurrentSpherecastRadiusLerp);

            // foreach (var item in RaySensorsList)
            // {
            //     item.MaxRayDegrees = Mathf.Lerp(MinMaxRayAngles.x, MinMaxRayAngles.y, CurrentRayAngleLerp);
            //     item.SphereCastRadius = Mathf.Lerp(MinMaxSpherecastRadius.x, MinMaxSpherecastRadius.y, CurrentSpherecastRadiusLerp);
            // }
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {

        if (position == Position.Goalie)
        {
            // Existential bonus for Goalies.
            AddReward(m_Existential);
        }
        else if (position == Position.Striker)
        {
            // Existential penalty for Strikers
            AddReward(-m_Existential);
        }
        MoveAgent(actionBuffers.ContinuousActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut.Clear();
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        //rotate
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[2] = 2;
        }
        //right
        if (Input.GetKey(KeyCode.E))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            discreteActionsOut[1] = 2;
        }
    }
    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = k_Power * m_KickPower;
        if (position == Position.Goalie)
        {
            force = k_Power;
        }
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.2f * m_BallTouch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
    }

}
