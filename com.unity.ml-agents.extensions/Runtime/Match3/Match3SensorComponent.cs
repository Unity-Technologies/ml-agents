using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Match3
{
    public class Match3SensorComponent : SensorComponent
    {
        public bool UseVectorObservations = true;

        public override ISensor CreateSensor()
        {
            var board = GetComponent<AbstractBoard>();
            return new Match3Sensor(board, UseVectorObservations);
        }

        public override int[] GetObservationShape()
        {
            var board = GetComponent<AbstractBoard>();
            if (board == null)
            {
                return System.Array.Empty<int>();
            }

            return UseVectorObservations ?
                new[] { board.Rows * board.Columns * board.NumCellTypes } :
                new[] { board.Rows, board.Columns, board.NumCellTypes };
        }
    }
}
