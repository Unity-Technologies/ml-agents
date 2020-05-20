using UnityEngine;

public class TennisArea : MonoBehaviour
{
    public GameObject ball;
    public GameObject agentA;
    public GameObject agentB;
    Rigidbody m_BallRb;
    HitWall m_BallScript;
    Vector3 m_Down = new Vector3(0f, -45f, 0f);

    // Use this for initialization
    void Start()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_BallScript = ball.GetComponent<HitWall>();
        MatchReset();
    }

    public void MatchReset()
    {
        var ballOut = Random.Range(12.5f, 14f);
        var flip = Random.Range(0, 2);
        var serve = 1f;
        if (flip == 0)
        {
            serve = -1f;
        }
        ball.transform.position = new Vector3(serve * ballOut, 1f, 0f) + transform.position;
        m_BallRb.velocity = new Vector3(serve * 5f, 10f, 0f);
        ball.transform.localScale = new Vector3(.5f, .5f, .5f);
        m_BallScript.ResetPoint();
    }

    void FixedUpdate()
    {
        m_BallRb.AddForce(m_Down);
        var rgV = m_BallRb.velocity;
        m_BallRb.velocity = new Vector3(Mathf.Clamp(rgV.x, -40f, 40f), Mathf.Min(rgV.y, 35f), rgV.z);
    }
}
