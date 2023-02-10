using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Demonstrations;
using UnityEngine;

public class SetDemoNameToAnimationName : MonoBehaviour
{
    void Awake()
    {
        var animator = GetComponentInChildren<Animator>();
        var runtimeController = animator.runtimeAnimatorController;
        var demoRecorder = GetComponentInChildren<DemonstrationRecorder>();
        if (runtimeController != null)
        {
            demoRecorder.DemonstrationName = runtimeController.name;
        }
    }
}
