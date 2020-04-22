using UnityEngine;
using UnityEngine.UI;
using MLAgents;
using MLAgents.Sensors;
using MLAgents.SideChannels;

public class TennisAgent : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public GameObject opponent;
    public bool invertX;
    public int score;
    public GameObject myArea;
    public float scale;

    Text m_TextComponent;
    Rigidbody m_AgentRb;
    Rigidbody m_BallRb;
    Rigidbody m_OpponentRb;
    HitWall m_BallScript;
    TennisArea m_Area;
    float m_InvertMult;
    FloatPropertiesChannel m_ResetParams;
    Vector3 down = new Vector3(0f, -100f, 0f);
    Vector3 zAxis = new Vector3(0f, 0f, 1f);
    const float k_Angle = 90f;
    const float k_MaxAngle = 145f;
    const float k_MinAngle = 35f;

    // Looks for the scoreboard based on the name of the gameObjects.
    // Do not modify the names of the Score GameObjects
    const string k_CanvasName = "Canvas";
    const string k_ScoreBoardAName = "ScoreA";
    const string k_ScoreBoardBName = "ScoreB";

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_OpponentRb = opponent.GetComponent<Rigidbody>();
        m_BallScript = ball.GetComponent<HitWall>();
        m_Area = myArea.GetComponent<TennisArea>();
        var canvas = GameObject.Find(k_CanvasName);
        GameObject scoreBoard;
        m_ResetParams = SideChannelUtils.GetSideChannel<FloatPropertiesChannel>();
        if (invertX)
        {
            scoreBoard = canvas.transform.Find(k_ScoreBoardBName).gameObject;
        }
        else
        {
            scoreBoard = canvas.transform.Find(k_ScoreBoardAName).gameObject;
        }
        m_TextComponent = scoreBoard.GetComponent<Text>();
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_InvertMult * (transform.position.x - myArea.transform.position.x));
        sensor.AddObservation(transform.position.y - myArea.transform.position.y);
        sensor.AddObservation(m_InvertMult * m_AgentRb.velocity.x);
        sensor.AddObservation(m_AgentRb.velocity.y);

        sensor.AddObservation(m_InvertMult * (ball.transform.position.x - myArea.transform.position.x));
        sensor.AddObservation(ball.transform.position.y - myArea.transform.position.y);
        sensor.AddObservation(m_InvertMult * m_BallRb.velocity.x);
        sensor.AddObservation(m_BallRb.velocity.y);

        sensor.AddObservation(m_InvertMult * (opponent.transform.position.x - myArea.transform.position.x));
        sensor.AddObservation(opponent.transform.position.y - myArea.transform.position.y);
        sensor.AddObservation(m_InvertMult * m_OpponentRb.velocity.x);
        sensor.AddObservation(m_OpponentRb.velocity.y);

        sensor.AddObservation(m_InvertMult * gameObject.transform.rotation.z);

        sensor.AddObservation(System.Convert.ToInt32(m_BallScript.lastFloorHit == HitWall.FloorHit.FloorHitUnset));
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * m_InvertMult;
        var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);
        var rotate = Mathf.Clamp(vectorAction[2], -1f, 1f) * m_InvertMult;
        
        var upward = 0.0f;
        if (moveY > 0.0 && transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            upward = moveY;
        }

        m_AgentRb.AddForce(new Vector3(moveX * 5f, upward * 10f, 0f), ForceMode.VelocityChange);

        // calculate angle between m_InvertMult * 35 and m_InvertMult * 145
        var angle = 55f * rotate + m_InvertMult * k_Angle;
        // maps inverse agents rotation into -35 to -145
        var rotateZ = angle - (gameObject.transform.rotation.eulerAngles.z - (1f - m_InvertMult) * 180f);
        Quaternion deltaRotation = Quaternion.Euler(zAxis * rotateZ);
        m_AgentRb.MoveRotation(m_AgentRb.rotation * deltaRotation);

        if (invertX && transform.position.x - transform.parent.transform.position.x < -m_InvertMult ||
            !invertX && transform.position.x - transform.parent.transform.position.x > -m_InvertMult)
        {
            transform.position = new Vector3(-m_InvertMult + transform.parent.transform.position.x,
                transform.position.y,
                transform.position.z);
        }
        var rgV = m_AgentRb.velocity;
        m_AgentRb.velocity = new Vector3(Mathf.Clamp(rgV.x, -20, 20), Mathf.Min(rgV.y, 10f), rgV.z);

        m_TextComponent.text = score.ToString();
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = Input.GetAxis("Horizontal");    // Racket Movement
        actionsOut[1] = Input.GetKey(KeyCode.Space) ? 1f : 0f;   // Racket Jumping
        actionsOut[2] = Input.GetAxis("Vertical");   // Racket Rotation
    }

    void OnCollisionEnter(Collision c)
    {
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.01f);
        }
    }

    void FixedUpdate()
    {   
        m_AgentRb.AddForce(down);
    }   

    public override void OnEpisodeBegin()
    {

        m_InvertMult = invertX ? -1f : 1f;
        var agentOutX = Random.Range(12f, 16f);
        var agentOutY = Random.Range(-1.5f, 0f);
        transform.position = new Vector3(-m_InvertMult * agentOutX, agentOutY, -1.8f) + transform.parent.transform.position;
        m_AgentRb.velocity = new Vector3(0f, 0f, 0f);
        SetResetParameters();
        if (m_InvertMult == 1f)
        {
            m_Area.MatchReset();
        }

    }

    public void SetRacket()
    {
        gameObject.transform.eulerAngles = new Vector3(
            gameObject.transform.eulerAngles.x,
            gameObject.transform.eulerAngles.y,
            m_InvertMult * k_Angle
        );
    }

    public void SetBall()
    {
        scale = m_ResetParams.GetPropertyWithDefault("scale", .5f);
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetRacket();
        SetBall();
    }
}
