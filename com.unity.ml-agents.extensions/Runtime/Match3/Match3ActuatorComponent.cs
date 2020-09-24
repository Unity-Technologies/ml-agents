using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Extensions.Match3
{
    public class Match3ActuatorComponent : ActuatorComponent
    {
        public bool ForceRandom = false;
        public override IActuator CreateActuator()
        {
            var randomSeed = 0;
            if (ForceRandom)
            {
                randomSeed = this.gameObject.GetInstanceID();
            }

            var board = GetComponent<AbstractBoard>();
            return new Match3Actuator(board, ForceRandom, randomSeed);
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

                var numMoves = Move.NumEdgeIndices(board.Rows, board.Columns);
                return ActionSpec.MakeDiscrete(numMoves);
            }
        }
    }
}
