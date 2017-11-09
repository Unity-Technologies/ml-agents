using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class TargetScript : MonoBehaviour {

    public RobotArmGameManagerContinuous GameManager;
    public UnityEvent OnHit;

    float touchDistance = 0f;

    private void Start()
    {
        touchDistance = (GameManager.Hand.transform.lossyScale.x + transform.lossyScale.x) * 0.5f;
    }

    private void FixedUpdate()
    {
        // If they intersect: Colliders aren't precise enough here
        if (Vector3.Distance(GameManager.Hand.transform.position, transform.position) <= touchDistance) {
            if (OnHit != null) OnHit.Invoke();
        }
    }
}
