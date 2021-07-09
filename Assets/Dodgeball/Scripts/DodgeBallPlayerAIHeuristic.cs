using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class DodgeBallPlayerAIHeuristic : MonoBehaviour
{
    [Header("TARGETS")]
    public Transform FollowTarget;
    public Transform LookAtTarget;

    [Header("VISION")] public bool canCurrentlySeeTarget;
    [Header("WALKING")]
    public bool RunTowardsTarget;
    public bool OnlyWalkIfCanSeeTarget = true;

    [Header("BODY ROTATION")]
    public float MaxRotationRate = 1;
    public float RandomRotationJitter = 1.5f;
    //    public LayerMask AgentLayer;
    public string TargetTag = "agent";
    public bool RotateTowardsTarget;

    [Header("SHOOTING")]
    public bool OnlyShootIfCanSeeTarget = true;


    private AgentCubeMovement moveController;
    private AgentHealth agentHealth;

    private ThrowBall ThrowController;
    public DodgeBallAgent m_Agent;
    // Start is called before the first frame update
    void Awake()
    {
        moveController = GetComponent<AgentCubeMovement>();
        agentHealth = GetComponent<AgentHealth>();
        ThrowController = GetComponent<ThrowBall>();
        m_Agent = GetComponent<DodgeBallAgent>();
    }

    void Update()
    {
        // if (!target)
        // {
        //     target = gameController.AITarget.transform;
        // }
        canCurrentlySeeTarget = false;
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, 50))
        {
            if (hit.transform == LookAtTarget) //simple vision
            {
                canCurrentlySeeTarget = true;
            }
        }
    }



    // Update is called once per frame
    void FixedUpdate()
    {
        if (agentHealth && agentHealth.Dead)
        {
            return;
        }
        // if (!target)
        // {
        //     target = gameController.AITarget.transform;
        // }
        Vector3 randomJitter = Random.insideUnitSphere * RandomRotationJitter;
        randomJitter.y = 0;
        Vector3 targetPos = FollowTarget.position + randomJitter;
        //        Vector3 dir = target.position - transform.position;
        Vector3 moveDir = targetPos - transform.position;
        Vector3 lookDir = (LookAtTarget.position - transform.position).normalized;
        if (RunTowardsTarget)
        {
            if (OnlyWalkIfCanSeeTarget && canCurrentlySeeTarget)
            {
                moveController.RunOnGround(moveDir.normalized);
            }
            else
            {
                moveController.RunOnGround(moveDir.normalized);

            }
        }

        if (RotateTowardsTarget)
        {
            moveController.RotateTowards(lookDir, MaxRotationRate);
        }

        if (OnlyShootIfCanSeeTarget && canCurrentlySeeTarget && m_Agent)
        {
            m_Agent.ThrowTheBall();
        }

        // if (OnlyShootIfCanSeeTarget && canCurrentlySeeTarget && ThrowController)
        // {
        //     ThrowController.Throw(ActiveBallsQueue.Peek());
        //
        //     multiGunController.Shoot();
        // }

    }
}
