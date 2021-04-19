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
            var seed = RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed + 1;
            return new IActuator[] { new Match3ExampleActuator(board, ForceHeuristic, ActuatorName, seed) };
        }
    }
}
