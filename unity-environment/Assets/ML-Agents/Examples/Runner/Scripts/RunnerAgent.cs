using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public interface IAgentTrigger
{
    void OnEnter(RunnerAgent agent);
}

public class RunnerAgent : Agent
{
    public enum AgentStatus
    {
        None,
        Killed, 
        Finished
    }

    private const int ColliderPhysicMask = (1 << 8);
    private const int KillzonePhysicMask = (1 << 9);
    private const int BonusPhysicMask = (1 << 10);

    [Header("Game specific")]
    [SerializeField]
    private float jumpVelocity = 6;

    [SerializeField]
    private float rollSpeed = 11.5f;

    [SerializeField]
    private Transform visualObject;

    [HideInInspector]
    public AgentStatus status;

    [HideInInspector]
    public int collectedBonus = 0;

    private Vector3 originalPosition;
    private Rigidbody rigidBody;
    private bool isOnFloor;

    public void Awake()
    {
        rigidBody = GetComponent<Rigidbody>();
        originalPosition = transform.position;
    }

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();

        state.Add(isOnFloor ? 1 : -1);

        ScanForKillzones(ref state, 2);
        state.Add(ScanForBonus());

        return state;
    }

    public override void AgentStep(float[] act)
    {
        switch (status)
        {
            case AgentStatus.Killed:
                {
                    reward = -1;
                    done = true;
                }
                break;
            case AgentStatus.Finished:
                {
                    reward = 1;
                    done = true;
                }
                break;
            default:
                {
                    if (isOnFloor)
                    {
                        if (act[0].Equals(1))
                        {
                            Jump();
                        }
                        else
                        {
                            reward += 0.05f;
                        }
                    }

                    reward += collectedBonus;
                    collectedBonus = 0;
                }
                break;
        }
    }

    public void FixedUpdate()
    {
        isOnFloor = ScanFloor();

        UpdateAnimation();
    }

    private void UpdateAnimation()
    {
        if (rigidBody.velocity.y > 0)
            visualObject.transform.localRotation = visualObject.transform.localRotation * Quaternion.AngleAxis(-rollSpeed, Vector3.forward);
        else
            visualObject.transform.localRotation = Quaternion.identity;
    }

    /// Search for kill zone in the ground
    /// Add the distance of each zone from the Agent and their sizes in states
    public void ScanForKillzones(ref List<float> states, int searchLimit)
    {
        float scanDistance = 10f;

        int killzoneFound = 0;
        float currentX = 0;
        while (currentX < scanDistance)
        {
            if (killzoneFound >= searchLimit)
                break;

            var currentPosition = transform.position + Vector3.right * currentX;

            RaycastHit hitInfo;
            Physics.Raycast(currentPosition, Vector3.down, out hitInfo, scanDistance, ColliderPhysicMask | KillzonePhysicMask);

            bool killZoneDetected = (hitInfo.collider != null && hitInfo.collider.tag == "killZone");

            if (killZoneDetected)
            {
                var raycastStart = hitInfo.point + Vector3.up * 0.5f;

                float killzoneStart = 0;
                float killzoneEnd = scanDistance;

                RaycastHit killzoneHitInfo;
                if (Physics.Raycast(raycastStart, -Vector3.right, out killzoneHitInfo, scanDistance, ColliderPhysicMask))
                {
                    killzoneStart = Math.Max(0, killzoneHitInfo.point.x - transform.position.x);
                }

                if (Physics.Raycast(raycastStart, Vector3.right, out killzoneHitInfo, scanDistance, ColliderPhysicMask))
                {
                    killzoneEnd = Math.Max(0, killzoneHitInfo.point.x - transform.position.x);

                    currentX = killzoneEnd + 1;
                }
                else
                {
                    currentX += 1f;
                }

                states.Add(killzoneStart / scanDistance);
                states.Add((killzoneEnd - killzoneStart) / scanDistance);
                killzoneFound++;

            }
            else
            {
                currentX += 1f;
            }
        }

        while (killzoneFound < searchLimit)
        {
            states.Add(-1);
            states.Add(-1);
            killzoneFound++;
        }
    }

    /// Search for next Bonus distance
    private float ScanForBonus()
    {
        float scanDistance = 5f;

        RaycastHit hitInfo;
        if (Physics.Raycast(new Vector3(transform.position.x + 0.5f, 0.75f, transform.position.z), Vector3.right, out hitInfo, scanDistance, BonusPhysicMask))
        {
            return hitInfo.distance / scanDistance;
        }

        return -1;
    }

    /// Detect if the cube is on the floor
    private bool ScanFloor()
    {
        return Physics.Raycast(transform.position + new Vector3(0.15f, 0, 0), Vector3.down, 0.3f, (1 << 8))
                            || Physics.Raycast(transform.position + new Vector3(-0.25f, 0, 0), Vector3.down, 0.3f, (1 << 8));
    }

    private void Jump()
    {
        rigidBody.velocity = new Vector3(0, jumpVelocity, 0);
    }


    public void Kill()
    {
        status = AgentStatus.Killed;
        visualObject.gameObject.SetActive(false);
    }

    public override void AgentReset()
    {
        Reinitialize();
    }

    public void Reinitialize()
    {
        visualObject.gameObject.SetActive(true);
        status = AgentStatus.None;
        rigidBody.velocity = Vector3.zero;
        transform.position = originalPosition;
    }


    #region Handle physic collisions
    public void OnCollisionEnter(Collision collision)
    {
        OnCollisionStay(collision);
    }

    public void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.tag == "killZone")
        {
            Kill();
        }
    }

    public void OnTriggerEnter(Collider collider)
    {
        if (collider.GetComponent<IAgentTrigger>() != null)
        {
            collider.GetComponent<IAgentTrigger>().OnEnter(this);
        }
    }
    #endregion

    #region Debug functions
    public void OnDrawGizmos()
    {
        if (done)
            return;

        int zoneDetection = 2;

        var collect = new List<float>();
        ScanForKillzones(ref collect, zoneDetection);

        Gizmos.color = Color.cyan;

        for (int i = 0; i < zoneDetection; i++)
        {
            if (collect[i * zoneDetection] > 0)
            {
                float zonePos = collect[i * 2] * 10;
                Gizmos.DrawLine(new Vector3(zonePos, -1.5f, transform.position.z - 0.4f), new Vector3(zonePos, 0.5f, transform.position.z - 0.4f));
                Gizmos.DrawLine(new Vector3(zonePos, -1.5f, transform.position.z + 0.4f), new Vector3(zonePos, 0.5f, transform.position.z + 0.4f));

                zonePos += collect[i * zoneDetection + 1] * 10;
                Gizmos.DrawLine(new Vector3(zonePos, -1.5f, transform.position.z - 0.4f), new Vector3(zonePos, 0.5f, transform.position.z - 0.4f));
                Gizmos.DrawLine(new Vector3(zonePos, -1.5f, transform.position.z + 0.4f), new Vector3(zonePos, 0.5f, transform.position.z + 0.4f));
            }
        }

        var bonusPosState = ScanForBonus();
        Gizmos.color = Color.yellow;
        if (bonusPosState > 0)
        {
            Gizmos.DrawLine(transform.position, new Vector3(transform.position.x + 0.5f + bonusPosState * 5, 0.75f, transform.position.z));
        }
    }
    #endregion
}
