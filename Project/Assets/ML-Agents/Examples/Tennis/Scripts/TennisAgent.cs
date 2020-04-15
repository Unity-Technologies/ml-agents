using UnityEngine;
using UnityEngine.UI;
using MLAgents;
using MLAgents.Sensors;
using MLAgents.SideChannels;

public class TennisAgent : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public bool invertX;
    public int score;
    public GameObject myArea;
    public float angle;
    public float scale;

    [HideInInspector]
    public float timePenalty = 0;

    Text m_TextComponent;
    Rigidbody m_AgentRb;
    Rigidbody m_BallRb;
    HitWall m_BallScript;
    TennisArea m_Area;
    float m_InvertMult;
    FloatPropertiesChannel m_ResetParams;
    float m_BallTouch;
    Vector3 down = new Vector3(0f, -100f, 0f);

    // Looks for the scoreboard based on the name of the gameObjects.
    // Do not modify the names of the Score GameObjects
    const string k_CanvasName = "Canvas";
    const string k_ScoreBoardAName = "ScoreA";
    const string k_ScoreBoardBName = "ScoreB";

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_BallRb = ball.GetComponent<Rigidbody>();
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

        sensor.AddObservation(m_InvertMult * gameObject.transform.rotation.z);
        
        sensor.AddObservation(System.Convert.ToInt32(m_BallScript.lastFloorHit == HitWall.FloorHit.FloorHitUnset));
        AddReward(m_BallTouch * (1f /  Vector3.Distance(ball.transform.position, transform.position)));
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * m_InvertMult;
        var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);
        var rotate = Mathf.Clamp(vectorAction[2], -1f, 1f) * m_InvertMult;

        if (moveY > 0.0 && transform.position.y - transform.parent.transform.position.y < 5f)
        {
            m_AgentRb.velocity = new Vector3(m_AgentRb.velocity.x, moveY * 20f, 0f);
        }

        m_AgentRb.velocity = new Vector3(moveX * 30f, m_AgentRb.velocity.y, 0f);

        m_AgentRb.transform.rotation = Quaternion.Euler(0f, -180f, 55f * rotate + m_InvertMult * 90f);

        if (invertX && transform.position.x - transform.parent.transform.position.x < -m_InvertMult ||
            !invertX && transform.position.x - transform.parent.transform.position.x > -m_InvertMult)
        {
            transform.position = new Vector3(-m_InvertMult + transform.parent.transform.position.x,
                transform.position.y,
                transform.position.z);
        }
        timePenalty += -1f / 3000f;

        m_TextComponent.text = score.ToString();
    }

    public override float[] Heuristic()
    {
        var action = new float[3];

        action[0] = Input.GetAxis("Horizontal");    // Racket Movement
        action[1] = Input.GetKey(KeyCode.Space) ? 1f : 0f;   // Racket Jumping
        action[2] = Input.GetAxis("Vertical");   // Racket Rotation  
        return action;
    }

    void OnCollisionEnter(Collision c)
    {
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.4f * m_BallTouch);
        }
    }

    void FixedUpdate()
    {   
        m_AgentRb.AddForce(down);
    }   

    public override void OnEpisodeBegin()
    {

        timePenalty = 0;
        m_BallTouch = SideChannelUtils.GetSideChannel<FloatPropertiesChannel>().GetPropertyWithDefault("ball_touch", 0);
        m_InvertMult = invertX ? -1f : 1f;
        if (m_InvertMult == 1f)
        {
            m_Area.MatchReset();
        }
        transform.position = new Vector3(-m_InvertMult * 14f, 1f, -1.8f) + transform.parent.transform.position;
        m_AgentRb.velocity = new Vector3(0f, 0f, 0f);
        SetResetParameters();
    }

    public void SetRacket()
    {
        angle = m_ResetParams.GetPropertyWithDefault("angle", 55);
        gameObject.transform.eulerAngles = new Vector3(
            gameObject.transform.eulerAngles.x,
            gameObject.transform.eulerAngles.y,
            m_InvertMult * angle
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
