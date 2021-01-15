using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Board;

namespace Unity.MLAgentsExamples
{
    public class Match3ActuatorComponent : BoardActuatorComponent
    {
        /// <inheritdoc/>
        public override IActuator CreateActuator()
        {
            var board = GetComponent<Match3Board>();
            var agent = GetComponentInParent<Agent>();
            var seed = board.RandomSeed == -1 ? gameObject.GetInstanceID() : board.RandomSeed + 1;
            return new Match3Actuator(board, ForceHeuristic, agent, ActuatorName, HeuristicQuality, seed);
        }
    }
}
