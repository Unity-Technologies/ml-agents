using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
{
    /// <summary>
    /// Sensor component for a Match3 game.
    /// </summary>
    [AddComponentMenu("ML Agents/Match 3 Sensor", (int)MenuGroup.Sensors)]
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
        public override ISensor[] CreateSensors()
        {
            var board = GetComponent<AbstractBoard>();
            var cellSensor = Match3Sensor.CellTypeSensor(board, ObservationType, SensorName + " (cells)");
            if (board.NumSpecialTypes > 0)
            {
                var specialSensor =
                    Match3Sensor.SpecialTypeSensor(board, ObservationType, SensorName + " (special)");
                return new ISensor[] { cellSensor, specialSensor };
            }
            else
            {
                return new ISensor[] { cellSensor };
            }
        }

    }
}
