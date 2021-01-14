using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Board
{
    /// <summary>
    /// Sensor component for a Match3 game.
    /// </summary>
    public class BoardSensorComponent : SensorComponent
    {
        /// <summary>
        /// Name of the generated BoardSensor object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName = "Match3 Sensor";

        /// <summary>
        /// Type of observation to generate.
        /// </summary>
        public BoardObservationType ObservationType = BoardObservationType.Vector;

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            var board = GetComponent<AbstractBoard>();
            return new BoardSensor(board, ObservationType, SensorName);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            var board = GetComponent<AbstractBoard>();
            if (board == null)
            {
                return System.Array.Empty<int>();
            }

            var specialSize = board.NumSpecialTypes == 0 ? 0 : board.NumSpecialTypes + 1;
            return ObservationType == BoardObservationType.Vector ?
                new[] { board.Rows * board.Columns * (board.NumCellTypes + specialSize) } :
                new[] { board.Rows, board.Columns, board.NumCellTypes + specialSize };
        }
    }
}
