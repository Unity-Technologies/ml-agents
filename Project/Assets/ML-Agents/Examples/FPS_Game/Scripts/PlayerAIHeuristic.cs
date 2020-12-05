using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class PlayerAIHeuristic : MonoBehaviour
{
    [Header("TARGET")]
    public Transform target;

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

    private MultiGunAlternating multiGunController;
    // Start is called before the first frame update
    void Start()
    {
        moveController = GetComponent<AgentCubeMovement>();
        agentHealth = GetComponent<AgentHealth>();
        multiGunController = GetComponent<MultiGunAlternating>();
    }

    void Update()
    {
        canCurrentlySeeTarget = false;
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, 50))
        {
            if (hit.transform == target) //simple vision
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
        Vector3 randomJitter = Random.insideUnitSphere * RandomRotationJitter;
        randomJitter.y = 0;
        Vector3 targetPos = target.position + randomJitter;
        //        Vector3 dir = target.position - transform.position;
        Vector3 dir = targetPos - transform.position;
        if (RunTowardsTarget)
        {
            if (OnlyWalkIfCanSeeTarget && canCurrentlySeeTarget)
            {
                moveController.RunOnGround(dir.normalized);
            }
        }

        if (RotateTowardsTarget)
        {
            moveController.RotateTowards(dir.normalized, MaxRotationRate);
        }

        if (OnlyShootIfCanSeeTarget && canCurrentlySeeTarget && multiGunController)
        {
            multiGunController.Shoot();
        }

    }
}
