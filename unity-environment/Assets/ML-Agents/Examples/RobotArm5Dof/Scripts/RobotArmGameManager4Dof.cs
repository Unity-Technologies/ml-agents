using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;



public class RobotArmGameManager4Dof : MonoBehaviour
{
    public int Score;
    public int targetHitValue = 1;

    public GameObject SpawnCenter;
    public Vector2 SpawnDistanceMinMax;
    public GameObject target;

    public RobotArmController4Dof ArmController;

    public UnityEvent OnTargetHit;
    public UnityEvent OnGameReset;

    public int CurrentStep = 0;


    // Use this for initialization
    void Start()
    {
        Time.captureFramerate = 60;
        ResetGame();
    }

    void FixedUpdate()
    {
        CurrentStep++;
    }

    public void ResetGame()
    {
        Score = 0;
        CurrentStep = 0;
        ArmController.Reset();
        MoveTarget();
        if (OnGameReset != null) OnGameReset.Invoke();
    }

    private void MoveTarget()
    {
        var dist = Random.Range(SpawnDistanceMinMax.x, SpawnDistanceMinMax.y);
        target.transform.position = GetPointOnUnitSphereCap(Quaternion.LookRotation(Vector3.up), dist) + transform.position + new Vector3(0f, 0.1f, 0f);
    }

    public void TargetHit()
    {
        MoveTarget();
        Score += targetHitValue;
        // if the target is hit, do something
        if (OnTargetHit != null) OnTargetHit.Invoke();
    }

    
    public static Vector3 GetPointOnUnitSphereCap(Quaternion targetDirection, float distance, float angle = 90)
    {
        var angleInRad = Random.Range(0.0f, angle) * Mathf.Deg2Rad;
        var PointOnCircle = (Random.insideUnitCircle.normalized) * Mathf.Sin(angleInRad);
        var V = new Vector3(PointOnCircle.x, PointOnCircle.y, Mathf.Cos(angleInRad));
        return (targetDirection * V) * distance;
    }
}
