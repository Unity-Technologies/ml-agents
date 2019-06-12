using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class CrawlerAcademy : Academy
{
    public override void InitializeAcademy()
    {
        Monitor.verticalOffset = 1f;
        Physics.defaultSolverIterations = 12;
        Physics.defaultSolverVelocityIterations = 12;
        Time.fixedDeltaTime = 0.01333f; // (75fps). default is .2 (60fps)
        Time.maximumDeltaTime = .15f; // Default is .33

        Physics.gravity = new Vector3(0, -resetParameters["gravity"], 0);
    }

    public override void AcademyReset()
    {
        Physics.gravity = new Vector3(0, -resetParameters["gravity"], 0);
    }

    public override void AcademyStep()
    {
    }
}
