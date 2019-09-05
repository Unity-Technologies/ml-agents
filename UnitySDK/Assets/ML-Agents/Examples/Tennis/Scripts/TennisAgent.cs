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
    private float m_InvertMul;
    private ResetParameters m_ResetParams;

    // Looks for the scoreboard based on the name of the gameObjects.
    // Do not modify the names of the Score GameObjects
    private const string k_CanvasName = "Canvas";
    private const string k_ScoreBoardAName = "ScoreA";
    private const string k_ScoreBoardBName = "ScoreB";

    protected override void InitializeAgent()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_BallRb = ball.GetComponent<Rigidbody>();
        var canvas = GameObject.Find(k_CanvasName);
        var academy = FindObjectOfType<Academy>();
        m_ResetParams = academy.resetParameters;
        var scoreBoard = invertX
            ? canvas.transform.Find(k_ScoreBoardBName).gameObject
            : canvas.transform.Find(k_ScoreBoardAName).gameObject;
        m_TextComponent = scoreBoard.GetComponent<Text>();
        SetResetParameters();
    }

    protected override void CollectObservations()
    {
        var position = transform.position;
        var myAreaPos = myArea.transform.position;
        AddVectorObs(m_InvertMul * (position.x - myAreaPos.x));
        AddVectorObs(position.y - myAreaPos.y);
        AddVectorObs(m_InvertMul * m_AgentRb.velocity.x);
        AddVectorObs(m_AgentRb.velocity.y);

        var ballPos = ball.transform.position;
        AddVectorObs(m_InvertMul * (ballPos.x - myAreaPos.x));
        AddVectorObs(ballPos.y - myAreaPos.y);
        AddVectorObs(m_InvertMul * m_BallRb.velocity.x);
        AddVectorObs(m_BallRb.velocity.y);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * m_InvertMul;
        var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);

        if (moveY > 0.5 && transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            m_AgentRb.velocity = new Vector3(m_AgentRb.velocity.x, 7f, 0f);
        }

        m_AgentRb.velocity = new Vector3(moveX * 30f, m_AgentRb.velocity.y, 0f);

        if (invertX && transform.position.x - transform.parent.transform.position.x < -m_InvertMul ||
            !invertX && transform.position.x - transform.parent.transform.position.x > -m_InvertMul)
        {
            Vector3 position;
            var myTransform = transform;
            position = new Vector3(-m_InvertMul + myTransform.parent.transform.position.x,
                (position = myTransform.position).y,
                position.z);
            myTransform.position = position;
        }

        m_TextComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        m_InvertMul = invertX ? -1f : 1f;

        var myTransform = transform;
        myTransform.position = new Vector3(-m_InvertMul * Random.Range(6f, 8f), -1.5f, 0f) + myTransform.parent.transform.position;
        m_AgentRb.velocity = new Vector3(0f, 0f, 0f);

        SetResetParameters();
    }

    void SetRacket()
    {
        angle = m_ResetParams["angle"];
        var o = gameObject;
        var eulerAngles = o.transform.eulerAngles;
        eulerAngles = new Vector3(
            eulerAngles.x,
            eulerAngles.y,
            m_InvertMul * angle
        );
        o.transform.eulerAngles = eulerAngles;
    }

    void SetBall()
    {
        scale = m_ResetParams["scale"];
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    void SetResetParameters()
    {
        SetRacket();
        SetBall();
    }
}
