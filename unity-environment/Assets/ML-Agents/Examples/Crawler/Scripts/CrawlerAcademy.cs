using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAcademy : Academy
{
    public bool randomTargetSpawnPos;
    public float targetSpawnRadius;
    public Transform target;

    public override void InitializeAcademy()
    {
        Monitor.verticalOffset = 1f;
        if(randomTargetSpawnPos)
        {
            GetRandomTargetPos();
        }
        Physics.defaultSolverIterations = 6;
        Physics.defaultSolverVelocityIterations = 6;
    }

    /// <summary>
    /// Moves target to a random position within specified radius.
    /// </summary>
    /// <returns>
    /// Move target to random position.
    /// </returns>
    public void GetRandomTargetPos()
    {
        Vector3 newTargetPos = Random.insideUnitSphere * targetSpawnRadius;
		newTargetPos.y = 5;
		target.position = newTargetPos;
    }

    public override void AcademyReset()
    {


    }

    public override void AcademyStep()
    {


    }

}
