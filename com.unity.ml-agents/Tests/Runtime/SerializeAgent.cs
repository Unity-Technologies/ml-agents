using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SerializeAgent : Agent
{
    public override float[] Heuristic()
    {
        return new[] {0f};
    }
}
