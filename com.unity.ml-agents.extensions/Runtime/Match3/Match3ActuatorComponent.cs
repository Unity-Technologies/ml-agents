using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Extensions.Match3
{
    public class Match3ActuatorComponent : ActuatorComponent
    {
        [FormerlySerializedAs("ForceRandom")]
        [Tooltip("Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.")]
        public bool ForceHeuristic = false;

        public override IActuator CreateActuator()
        {
            var board = GetComponent<AbstractBoard>();
            var agent = GetComponentInParent<Agent>();
            return new Match3Actuator(board, ForceHeuristic, agent);
        }

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
