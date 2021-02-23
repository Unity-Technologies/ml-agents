using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Sensor component for a Match3 game.
    /// </summary>
    public class Match3SensorComponent : SensorComponent
    {
        /// <summary>
        /// Name of the generated Match3Sensor object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName = "Match3 Sensor";

        /// <summary>
        /// Type of observation to generate.
        /// </summary>
        public Match3ObservationType ObservationType = Match3ObservationType.Vector;

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            var board = GetComponent<AbstractBoard>();
            return new Match3Sensor(board, ObservationType, SensorName);
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
            return ObservationType == Match3ObservationType.Vector ?
                new[] { board.Rows * board.Columns * (board.NumCellTypes + specialSize) } :
                new[] { board.Rows, board.Columns, board.NumCellTypes + specialSize };
        }
    }
}
