using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class RobotArmGameManager : MonoBehaviour
{
    public float score;
    public float targetHitValue = 1f;

    public GameObject SpawnCenter;
    public float SpawnDistance;
    public GameObject target;

    public GameObject Rotator1;
    public GameObject Bender1;
    public GameObject Hand;

    public RobotArmController ArmController;

    public UnityEvent OnTargetHit;
    public UnityEvent OnGameReset;

    // Use this for initialization
    void Start()
    {
        Time.captureFramerate = 60;
        ResetGame();
    }

    public void ResetGame()
    {
        score = 0;
        Rotator1.transform.rotation = Quaternion.identity;
        Bender1.transform.rotation = Quaternion.identity;
        MoveTarget();
        if (OnGameReset != null) OnGameReset.Invoke();
    }

    private void MoveTarget()
    {
        target.transform.position = GetPointOnUnitSphereCap(Quaternion.LookRotation(Vector3.up), SpawnDistance) + transform.position + new Vector3(0f, 0.1f, 0f);
    }

    public void TargetHit()
    {
        MoveTarget();
        score += targetHitValue;
        // if the target is hit, do something
        if (OnTargetHit != null) OnTargetHit.Invoke();
    }

    //C# version
    public static Vector3 GetPointOnUnitSphereCap(Quaternion targetDirection, float distance, float angle = 90)
    {
        var angleInRad = Random.Range(0.0f, angle) * Mathf.Deg2Rad;
        var PointOnCircle = (Random.insideUnitCircle.normalized) * Mathf.Sin(angleInRad);
        var V = new Vector3(PointOnCircle.x, PointOnCircle.y, Mathf.Cos(angleInRad));
        return (targetDirection * V) * distance;
    }
}
