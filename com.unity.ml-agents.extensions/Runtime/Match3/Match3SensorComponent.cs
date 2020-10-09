using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Match3
{
    public class Match3SensorComponent : SensorComponent
    {
        public Match3ObservationType ObservationType = Match3ObservationType.Vector;

        public override ISensor CreateSensor()
        {
            var board = GetComponent<AbstractBoard>();
            return new Match3Sensor(board, ObservationType);
        }

        public override int[] GetObservationShape()
        {
            var board = GetComponent<AbstractBoard>();
            if (board == null)
            {
                return System.Array.Empty<int>();
            }

            var specialSize = board.NumSpecialTypes == 0 ? 0 : board.NumSpecialTypes + 1;
            return ObservationType == Match3ObservationType.Vector ?
                new[] { board.Rows * board.Columns * (board.NumCellTypes + specialSize) } :
                new[] { board.Rows, board.Columns, board.NumCellTypes + specialSize };
        }
    }
}
