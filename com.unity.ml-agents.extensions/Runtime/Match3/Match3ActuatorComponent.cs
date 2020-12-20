using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Actuator component for a Match 3 game. Generates a Match3Actuator at runtime.
    /// </summary>
    public class Match3ActuatorComponent : ActuatorComponent
    {
        /// <summary>
        /// Name of the generated Match3Actuator object.
        /// Note that changing this at runtime does not affect how the Agent sorts the actuators.
        /// </summary>
        public string ActuatorName = "Match3 Actuator";

        /// <summary>
        /// Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.
        /// </summary>
        [FormerlySerializedAs("ForceRandom")]
        [Tooltip("Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.")]
        public bool ForceHeuristic;

        /// <inheritdoc/>
        public override IActuator CreateActuator()
        {
            var board = GetComponent<AbstractBoard>();
            var agent = GetComponentInParent<Agent>();
            return new Match3Actuator(board, ForceHeuristic, agent, ActuatorName);
        }

        /// <inheritdoc/>
        public override ActionSpec ActionSpec
        {
            get
            {
                var board = GetComponent<AbstractBoard>();
                if (board == null)
                {
                    return ActionSpec.MakeContinuous(0);
                }

                var numMoves = Move.NumPotentialMoves(board.Rows, board.Columns);
                return ActionSpec.MakeDiscrete(numMoves);
            }
        }
    }
}
