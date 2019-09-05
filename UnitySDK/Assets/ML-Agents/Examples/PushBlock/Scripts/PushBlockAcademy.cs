//Every scene needs an academy script.
//Create an empty gameObject and attach this script.
//The brain needs to be a child of the Academy gameObject.

using UnityEngine;
using MLAgents;

public class PushBlockAcademy : Academy
{
    /// <summary>
    /// The "walking speed" of the agents in the scene.
    /// </summary>
    public float agentRunSpeed;

    /// <summary>
    /// The spawn area margin multiplier.
    /// ex: .9 means 90% of spawn area will be used.
    /// .1 margin will be left (so players don't spawn off of the edge).
    /// The higher this value, the longer training time required.
    /// </summary>
    public float spawnAreaMarginMultiplier;

    /// <summary>
    /// When a goal is scored the ground will switch to this
    /// material for a few seconds.
    /// </summary>
    public Material goalScoredMaterial;
}
