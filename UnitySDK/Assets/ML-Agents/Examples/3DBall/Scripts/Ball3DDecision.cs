using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class Ball3DDecision : Decision
{
    public float rotationSpeed = 2f;

    public override float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        if (brainParameters.vectorActionSpaceType == SpaceType.Continuous)
        {
            var act = new List<float>
            {
                vectorObs[5] * rotationSpeed,
                -vectorObs[7] * rotationSpeed
            };

            // state[5] is the velocity of the ball in the x orientation.
            // We use this number to control the Platform's z axis rotation speed,
            // so that the Platform is tilted in the x orientation correspondingly.

            // state[7] is the velocity of the ball in the z orientation.
            // We use this number to control the Platform's x axis rotation speed,
            // so that the Platform is tilted in the z orientation correspondingly.

            return act.ToArray();
        }

        // If the vector action space type is discrete, then we don't do anything.
        return new[] { 1f };
    }

    public override List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}
