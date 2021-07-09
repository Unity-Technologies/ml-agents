using UnityEngine;
using Unity.MLAgents;

public class DodgeBall : MonoBehaviour
{

    public bool inPlay;

    [HideInInspector]
    public Rigidbody rb;

    public Collider BallCollider;

    public int TeamToIgnore;
    public DodgeBallAgent thrownBy;

    private Material m_BallMat;

    [Header("COLOR FLASHING")] public int FlashFrequency = 3; //The rate the ball should flash based on frames;
    Color m_PrimaryColor;
    public Color FlashColor = Color.white;

    private Vector3 m_ResetPosition;

    private EnvironmentParameters m_ResetParams;

    private TrailRenderer m_TrailRenderer;
    public void SetResetPosition(Vector3 position)
    {
        m_ResetPosition = position;
        m_TrailRenderer.Clear();
    }

    void Awake()
    {
        thrownBy = null;
        m_BallMat = BallCollider.gameObject.GetComponent<MeshRenderer>().material;
        m_TrailRenderer = GetComponentInChildren<TrailRenderer>();
        m_PrimaryColor = m_BallMat.color;
        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }


    private void OnEnable()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (inPlay)
        {
            if (FlashFrequency > 0 && Time.frameCount % FlashFrequency == 0)
            {
                m_BallMat.color = m_BallMat.color == m_PrimaryColor ? FlashColor : m_PrimaryColor;
            }
        }
    }

    //Set ball to either a pickup item or an active ball that is in play
    //inPlay = true means the ball can hurt other players
    //ignoreTeam = the TeamID to ignore. *friendly fire mechanic
    public void BallIsInPlay(bool p, int ignoreTeam = -1)
    {
        if (p)
        {
            TagBallAs("dodgeBallActive");
        }
        else
        {
            TagBallAs("dodgeBallPickup");
            m_BallMat.color = m_PrimaryColor;
        }
        inPlay = p;
        TeamToIgnore = ignoreTeam;
    }

    void TagBallAs(string tag)
    {
        gameObject.tag = tag;
        BallCollider.gameObject.tag = tag;
    }

    private void OnCollisionEnter(Collision col)
    {
        //IF NOT MY TEAM
        //PLAYER GOES TO TIMEOUT
        if (col.gameObject.CompareTag("ground"))
        {
            BallIsInPlay(false);
            thrownBy = null;
        }
    }
}
