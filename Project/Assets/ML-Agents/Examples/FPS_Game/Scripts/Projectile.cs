using UnityEngine;


//This script handles the logic to determine whether an active projectile
//...should be remain active after it has been shot by an enemy
public class Projectile : MonoBehaviour
{
    public float aliveTime;
    [HideInInspector] public Rigidbody rb;
    //    private ObstacleTowerAgent agent;
    public bool selfDestructNow;
    public float maxTimeToLive = 3;
    public float pauseCollisionDetectionWaitTime = .5f;
    //    [HideInInspector] public ShootProjectiles projectileController;
    public PoolGameObjects impactParticlePool;
    public TrailRenderer trailRend;
    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        //        agent = FindObjectOfType<ObstacleTowerAgent>();
        impactParticlePool = FindObjectOfType<PoolGameObjects>();
    }


    void OnEnable()
    {
        if (!rb)
        {
            rb = GetComponent<Rigidbody>();
        }

        aliveTime = 0;
        selfDestructNow = false;
        if (trailRend)
        {
            trailRend.Clear(); ;
        }
        //        if (agent)
        //        {
        //            agent.CompletedFloorAction += SelfDestruct;
        //        }
    }


    void OnDisable()
    {
        aliveTime = 0;
        //        if (agent)
        //        {
        //            agent.CompletedFloorAction -= SelfDestruct;
        //        }
    }

    //Turn the projectile off
    void SelfDestruct()
    {
        gameObject.SetActive(false);
        //        rb.velocity = Vector3.zero;
        //        rb.angularVelocity = Vector3.zero;
    }

    void FixedUpdate()
    {
        //        if (
        //            (agent && agent.IsDone()) //if the agent is done projectiles can die
        //            || aliveTime > maxTimeToLive //we lived too long. time to die
        //        )
        if (aliveTime > maxTimeToLive) //we lived too long. time to die
        {
            selfDestructNow = true;
        }

        if (selfDestructNow)
        {
            SelfDestruct();
        }

        aliveTime += Time.fixedDeltaTime;
    }


    void OnCollisionEnter(Collision col)
    {
        if (aliveTime > pauseCollisionDetectionWaitTime)
        {
            selfDestructNow = true;
            SelfDestruct();

        }

        foreach (var item in impactParticlePool.poolList)
        {
            if (!item.gameObject.activeInHierarchy)
            {
                item.transform.position = rb.transform.position;
                item.transform.rotation = Quaternion.LookRotation(col.contacts[0].normal);
                // print(col.contacts[0].normal);
                // item.transform.rotation = Quaternion.Euler(col.contacts[0].normal);
                item.gameObject.SetActive(true);
                return;
            }
        }
        //        print(col.gameObject.name);
    }
}
