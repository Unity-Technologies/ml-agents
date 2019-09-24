using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class TennisAgent : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public bool invertX;
    public int score;
    public GameObject myArea;
    public float angle;
    public float scale;

    private Text m_TextComponent;
    private Rigidbody m_AgentRb;
    private Rigidbody m_BallRb;
    private float m_InvertMult;
    private ResetParameters m_ResetParams;

    // Looks for the scoreboard based on the name of the gameObjects.
    // Do not modify the names of the Score GameObjects
    private const string k_CanvasName = "Canvas";
    private const string k_ScoreBoardAName = "ScoreA";
    private const string k_ScoreBoardBName = "ScoreB";

    public override void InitializeAgent()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_BallRb = ball.GetComponent<Rigidbody>();
        var canvas = GameObject.Find(k_CanvasName);
        GameObject scoreBoard;
        var academy = FindObjectOfType<Academy>();
        m_ResetParams = academy.resetParameters;
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

    public override void CollectObservations()
    {
        AddVectorObs(m_InvertMult * (transform.position.x - myArea.transform.position.x));
        AddVectorObs(transform.position.y - myArea.transform.position.y);
        AddVectorObs(m_InvertMult * m_AgentRb.velocity.x);
        AddVectorObs(m_AgentRb.velocity.y);

        AddVectorObs(m_InvertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
        AddVectorObs(m_InvertMult * m_BallRb.velocity.x);
        AddVectorObs(m_BallRb.velocity.y);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * m_InvertMult;
        var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);

        if (moveY > 0.5 && transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            m_AgentRb.velocity = new Vector3(m_AgentRb.velocity.x, 7f, 0f);
        }

        m_AgentRb.velocity = new Vector3(moveX * 30f, m_AgentRb.velocity.y, 0f);

        if (invertX && transform.position.x - transform.parent.transform.position.x < -m_InvertMult ||
            !invertX && transform.position.x - transform.parent.transform.position.x > -m_InvertMult)
        {
            transform.position = new Vector3(-m_InvertMult + transform.parent.transform.position.x,
                transform.position.y,
                transform.position.z);
        }

        m_TextComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        m_InvertMult = invertX ? -1f : 1f;

        transform.position = new Vector3(-m_InvertMult * Random.Range(6f, 8f), -1.5f, -3.5f) + transform.parent.transform.position;
        m_AgentRb.velocity = new Vector3(0f, 0f, 0f);

        SetResetParameters();
    }

    public void SetRacket()
    {
        angle = m_ResetParams["angle"];
        gameObject.transform.eulerAngles = new Vector3(
            gameObject.transform.eulerAngles.x,
            gameObject.transform.eulerAngles.y,
            m_InvertMult * angle
        );
    }

    public void SetBall()
    {
        scale = m_ResetParams["scale"];
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetRacket();
        SetBall();
    }
}
