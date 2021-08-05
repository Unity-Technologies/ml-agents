using UnityEngine;

public class SimpleNPC : MonoBehaviour
{

    public Transform target;

    private Rigidbody rb;
    public bool KeyCarrier = false;

    private DungeonEscapeEnvController m_GameController;
    public float walkSpeed = 1;
    // public ForceMode walkForceMode;
    private Vector3 dirToGo;

    // private Vector3 m_StartingPos;
    // Start is called before the first frame update

    public void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        m_GameController = GetComponentInParent<DungeonEscapeEnvController>();
    }
    void Awake()
    {
        Initialize();
        // m_StartingPos = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
    }


    void OnCollisionEnter(Collision col)
    {
        if (KeyCarrier)
        {

            if (col.transform.CompareTag("portal"))
            {
                m_GameController.BaddieTouchedBlock();
            }
        }
    }

    void FixedUpdate()
    {
        Vector3 goToward;
        if (!KeyCarrier)
        {
            goToward = m_GameController.GetNearestAgent(transform.position);
        }
        else
        {
            goToward = target.position;
        }

        dirToGo = goToward - transform.position;
        dirToGo.y = 0;
        rb.rotation = Quaternion.LookRotation(dirToGo);
        // rb.AddForce(dirToGo.normalized * walkSpeed * Time.fixedDeltaTime, walkForceMode);
        // rb.MovePosition(rb.transform.TransformDirection(Vector3.forward * walkSpeed * Time.deltaTime));
        // rb.MovePosition(rb.transform.TransformVector() (Vector3.forward * walkSpeed * Time.deltaTime));
        rb.MovePosition(transform.position + transform.forward * walkSpeed * Time.deltaTime);
    }

    public void SetRandomWalkSpeed()
    {
        walkSpeed = Random.Range(1f, 7f);
    }
}
