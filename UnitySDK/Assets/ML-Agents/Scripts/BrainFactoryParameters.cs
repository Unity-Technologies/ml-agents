using UnityEngine.Serialization;
using UnityEngine;
using Barracuda;

namespace MLAgents
{
    [System.Serializable]
    public class BrainFactoryParameters
    {

        [SerializeField] public string behaviorName = "MyBehavior";
        [SerializeField] public BrainParameters brainParameters;
        [SerializeField] public NNModel model;

        [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
            "(This field is not applicable for training).")]
        [SerializeField] public InferenceDevice inferenceDevice = InferenceDevice.CPU;

        [SerializeField] public bool useHeuristic;
    }
}
