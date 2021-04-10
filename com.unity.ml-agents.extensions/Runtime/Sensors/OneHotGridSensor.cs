using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor with one-hot observations.
    /// </summary>
    public class OneHotGridSensor : GridSensorBase
    {
        public OneHotGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridNum,
            string[] detectableTags,
            SensorCompressionType compression
        ) : base(name, cellScale, gridNum, detectableTags, compression)
        {

        }

        /// <inheritdoc/>
        protected override int GetCellObservationSize()
        {
            return DetectableTags == null ? 0 : DetectableTags.Length;
        }

        /// <inheritdoc/>
        protected override bool IsDataNormalized()
        {
            return true;
        }

        /// <inheritdoc/>
        protected internal override bool ProcessAllCollidersInCell()
        {
            return false;
        }

        /// <summary>
        /// Get the one-hot representation of the detected game object's tag.
        /// </summary>
        /// <param name="detectedObject">The game object that was detected within a certain cell</param>
        /// <param name="tagIndex">The index of the detectedObject's tag in the DetectableObjects list</param>
        /// <param name="dataBuffer">The buffer to write the observation values.
        ///         The buffer size is configured by <seealso cref="GetCellObservationSize"/>.
        /// </param>
        protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
        {
            dataBuffer[tagIndex] = 1;
        }
    }
}
