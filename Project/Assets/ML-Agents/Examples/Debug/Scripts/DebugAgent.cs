using UnityEngine;
using MLAgents;

namespace MLAgentsExamples
{
    public class DebugAgent : Agent
    {
        public override void CollectObservations(VectorSensor sensor)
        {
            for (var i = 0; i < 88; i++)
            {
                sensor.AddObservation(0.0f);
            }
        }

        public override void AgentAction(float[] vectorAction)
        {

        }
    }
}
