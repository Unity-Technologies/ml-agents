using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3ExampleActuatorComponent : Match3ActuatorComponent
    {
        /// <inheritdoc/>
#pragma warning disable 672
        public override IActuator CreateActuator()
#pragma warning restore 672
        {
            var board = GetComponent<Match3Board>();
            var agent = GetComponentInParent<Agent>();
            var seed = RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed + 1;
            return new Match3ExampleActuator(board, ForceHeuristic, agent, ActuatorName, seed);
        }
    }
}
