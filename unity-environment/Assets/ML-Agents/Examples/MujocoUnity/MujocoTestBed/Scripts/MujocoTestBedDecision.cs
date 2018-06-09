using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;

public class MujocoTestBedDecision : MonoBehaviour, Decision
{
    Brain _brain;
    [Tooltip("Action applied to each motor")]
    /**< \brief Edit to manually test each motor (+1/-1)*/
    public float[] Actions;

    [Tooltip("Apply a random number to each action each framestep")]
    /**< \brief Apply a random number to each action each framestep*/
    public bool ApplyRandomActions = false;


    // [Tooltip("Lock the top most element")]
    // /**< \brief Lock the top most element*/
    // public bool FreezeTop = false;
    // bool _lastFreezeTop;

    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        if (ApplyRandomActions) {
            for (int i = 0; i < Actions.Length; i++)
                Actions[i] = UnityEngine.Random.value * 2 - 1;
        }
        return Actions;
    }

    void Start()
    {
        _brain = this.GetComponent<Brain>();
        Actions = Enumerable.Repeat<float>(0f, _brain.brainParameters.vectorActionSize).ToArray();
    }

    public List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}
