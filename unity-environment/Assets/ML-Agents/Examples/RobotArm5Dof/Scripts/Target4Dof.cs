using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class Target4Dof : MonoBehaviour
{
    public RobotArmGameManager4Dof GameManager;
    public UnityEvent OnHit;
    public GameObject HitCenter;

    public float touchDistance = 0f;

    private void FixedUpdate()
    {
        // If they intersect: Colliders aren't precise enough here
        if (Vector3.Distance(HitCenter.transform.position, transform.position) <= touchDistance)
        {
            if (OnHit != null) OnHit.Invoke();
        }
    }
}
