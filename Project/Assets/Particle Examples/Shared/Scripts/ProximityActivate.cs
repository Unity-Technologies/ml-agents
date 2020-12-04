using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ProximityActivate : MonoBehaviour
{

    public Transform distanceActivator, lookAtActivator;
    public float distance;
    public Transform activator;
    public bool activeState = false;
    public CanvasGroup target;
    public bool lookAtCamera = true;
    public bool enableInfoPanel = false;
    public GameObject infoIcon;

    float alpha;
    public CanvasGroup infoPanel;
    Quaternion originRotation, targetRotation;

    void Start()
    {
        originRotation = transform.rotation;
        alpha = activeState ? 1 : -1;
        if (activator == null) activator = Camera.main.transform;
        infoIcon.SetActive(infoPanel != null);
    }

    bool IsTargetNear()
    {
        var distanceDelta = distanceActivator.position - activator.position;
        if (distanceDelta.sqrMagnitude < distance * distance)
        {
            if (lookAtActivator != null)
            {
                var lookAtActivatorDelta = lookAtActivator.position - activator.position;
                if (Vector3.Dot(activator.forward, lookAtActivatorDelta.normalized) > 0.95f)
                    return true;
            }
            var lookAtDelta = target.transform.position - activator.position;
            if (Vector3.Dot(activator.forward, lookAtDelta.normalized) > 0.95f)
                return true;
        }
        return false;
    }

    void Update()
    {
        if (!activeState)
        {
            if (IsTargetNear())
            {
                alpha = 1;
                activeState = true;
            }
        }
        else
        {
            if (!IsTargetNear())
            {
                alpha = -1;
                activeState = false;
                enableInfoPanel = false;
            }
        }
        target.alpha = Mathf.Clamp01(target.alpha + alpha * Time.deltaTime);
        if (infoPanel != null)
        {
            if (Input.GetKeyDown(KeyCode.Space))
                enableInfoPanel = !enableInfoPanel;
            infoPanel.alpha = Mathf.Lerp(infoPanel.alpha, Mathf.Clamp01(enableInfoPanel ? alpha : 0), Time.deltaTime * 10);
        }
        if (lookAtCamera)
        {
            if (activeState)
                targetRotation = Quaternion.LookRotation(activator.position - transform.position);
            else
                targetRotation = originRotation;
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime);
        }
    }

}
