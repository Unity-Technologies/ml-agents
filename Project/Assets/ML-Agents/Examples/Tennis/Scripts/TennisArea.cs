using UnityEngine;

public class TennisArea : MonoBehaviour
{
    public GameObject ball;
    public GameObject agentA;
    public GameObject agentB;
    Rigidbody m_BallRb;
    HitWall m_BallScript;
    Vector3 down = new Vector3(0f, -45f, 0f);

    // Use this for initialization
    void Start()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_BallScript = ball.GetComponent<HitWall>();
        MatchReset();
    }

    public void MatchReset()
    {
        var ballOut = 14f;
        var flip = Random.Range(0, 2);
        if (flip == 0)
        {
            ball.transform.position = new Vector3(-ballOut, 10f, 0f) + transform.position;
        }
        else
        {
            ball.transform.position = new Vector3(ballOut, 10f, 0f) + transform.position;
        }
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.localScale = new Vector3(.5f, .5f, .5f);
        m_BallScript.ResetPoint();
    }

    void FixedUpdate()
    {
        m_BallRb.AddForce(down);
        var rgV = m_BallRb.velocity;
        m_BallRb.velocity = new Vector3(Mathf.Clamp(rgV.x, -40f, 40f), Mathf.Clamp(rgV.y, -45f, 45f), rgV.z);
    }
}
