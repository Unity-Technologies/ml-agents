using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Actuator component for a Match3 game. Generates a Match3Actuator at runtime.
    /// </summary>
    [AddComponentMenu("ML Agents/Match 3 Actuator", (int)MenuGroup.Actuators)]
    public class Match3ActuatorComponent : ActuatorComponent
    {
        /// <summary>
        /// Name of the generated Match3Actuator object.
        /// Note that changing this at runtime does not affect how the Agent sorts the actuators.
        /// </summary>
        public string ActuatorName = "Match3 Actuator";

        /// <summary>
        /// A random seed used to generate a board, if needed.
        /// </summary>
        public int RandomSeed = -1;

        /// <summary>
        /// Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.
        /// </summary>
        [FormerlySerializedAs("ForceRandom")]
        [Tooltip("Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.")]
        public bool ForceHeuristic;

        /// <inheritdoc/>
#pragma warning disable 672
        public override IActuator CreateActuator()
#pragma warning restore 672
        {
            var board = GetComponent<AbstractBoard>();
            var agent = GetComponentInParent<Agent>();
            var seed = RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed + 1;
            return new Match3Actuator(board, ForceHeuristic, seed, agent, ActuatorName);
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
