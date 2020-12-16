using UnityEngine;

namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// A component for BufferSensor.
    /// </summary>
    [AddComponentMenu("ML Agents/Buffer Sensor", (int)MenuGroup.Sensors)]
    public class BufferSensorComponent : SensorComponent
    {
        public int ObservableSize;
        public int MaxNumObservables;

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            return new BufferSensor(ObservableSize, MaxNumObservables);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { MaxNumObservables, ObservableSize };
        }
    }
}
