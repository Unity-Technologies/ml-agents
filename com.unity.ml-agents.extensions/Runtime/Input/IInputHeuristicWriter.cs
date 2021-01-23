using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IInputHeuristicWriter
    {
        void WriteToHeuristic(InputAction action,
            in ActionBuffers actionBuffers,
            int continuousOffset,
            int discreteOffset);
    }
}
