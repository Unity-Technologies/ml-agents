using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Integrations.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3ExampleActuatorComponent : Match3ActuatorComponent
    {
        /// <inheritdoc/>
        public override IActuator[] CreateActuators()
        {
            var board = GetComponent<Match3Board>();
            var seed = RandomSeed == -1 ? CreateNewSeed() : RandomSeed + 1;
            return new IActuator[] { new Match3ExampleActuator(board, ForceHeuristic, ActuatorName, seed) };
        }

        int CreateNewSeed()
        {
#if UNITY_6000_3_OR_NEWER
            return gameObject.GetEntityId().GetHashCode();
#else
            return gameObject.GetInstanceID();
#endif
        }
    }
}
