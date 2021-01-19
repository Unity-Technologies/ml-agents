using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3ExampleActuatorComponent : Match3ActuatorComponent
    {
        /// <inheritdoc/>
        public override IActuator CreateActuator()
        {
            var board = GetComponent<Match3Board>();
            var agent = GetComponentInParent<Agent>();
            var seed = RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed + 1;
            board.RandomSeed = seed;
            return new Match3ExampleActuator(board, ForceHeuristic, agent, ActuatorName, seed);
        }
    }
}
