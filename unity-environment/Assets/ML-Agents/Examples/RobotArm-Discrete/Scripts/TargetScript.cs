using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class TargetScript : MonoBehaviour {

    public UnityEvent OnHit;

    private void OnTriggerEnter(Collider other)
    {
        if (OnHit != null) OnHit.Invoke();
    }
}
