using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Extensions.Board
{
    /// <summary>
    /// Actuator component for a Board game. Generates a BoardActuator at runtime.
    /// </summary>
    public class BoardActuatorComponent : ActuatorComponent
    {
        /// <summary>
        /// Name of the generated BoardActuator object.
        /// Note that changing this at runtime does not affect how the Agent sorts the actuators.
        /// </summary>
        public string ActuatorName = "Board Actuator";

        /// <summary>
        /// Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.
        /// </summary>
        [FormerlySerializedAs("ForceRandom")]
        [Tooltip("Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.")]
        public bool ForceHeuristic;

        public HeuristicQuality HeuristicQuality;

        /// <inheritdoc/>
        public override IActuator CreateActuator()
        {
            var board = GetComponent<AbstractBoard>();
            var agent = GetComponentInParent<Agent>();
            var seed = board.RandomSeed == -1 ? gameObject.GetInstanceID() : board.RandomSeed + 1;
            return new BoardActuator(board, ForceHeuristic, HeuristicQuality, seed, agent, ActuatorName);
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
