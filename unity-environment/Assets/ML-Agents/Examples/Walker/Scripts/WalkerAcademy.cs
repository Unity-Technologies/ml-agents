using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class WalkerAcademy : Academy
{
    public override void InitializeAcademy()
    {
        Monitor.verticalOffset = 1f;
        
        // We increase the Physics solver iterations in order to
        // make walker joint calculations more accurate.
        Physics.defaultSolverIterations = 12;
        Physics.defaultSolverVelocityIterations = 12;
    }

    public override void AcademyReset()
    {

    }

    public override void AcademyStep()
    {

    }
}
