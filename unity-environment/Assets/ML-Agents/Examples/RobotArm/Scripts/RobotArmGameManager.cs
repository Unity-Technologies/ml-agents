using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RobotArmGameManager : MonoBehaviour
{
    public float timeTaken;
    public float score;
    public float distanceToTarget;
    public float gameLength = 30f;
    public float targetHitValue = 1f;

    public UnityFloatEvent OnBallReleased;
    public GameObject SpawnCenter;
    public float SpawnDistance;
    public GameObject target;

    // Use this for initialization
    void Start()
    {
        Time.captureFramerate = 60;
        ResetGame();
    }

    public void ResetGame()
    {
        score = 0;
        timeTaken = 0f;
        MoveTarget();
    }

    private void Update()
    {
        timeTaken += Time.deltaTime;
        if (timeTaken > gameLength) ResetGame();
    }

    private void MoveTarget()
    {
        target.transform.position = GetPointOnUnitSphereCap(Quaternion.LookRotation(Vector3.up), SpawnDistance);
    }

    public void TargetHit()
    {
        score += targetHitValue;
        MoveTarget();
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
