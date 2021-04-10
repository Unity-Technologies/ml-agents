using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor that counts the number of detctable objects.
    /// </summary>
    public class CountingGridSensor : GridSensorBase
    {
        public CountingGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridNum,
            string[] detectableTags,
            SensorCompressionType compression
        ) : base(name, cellScale, gridNum, detectableTags, compression)
        {
            CompressionType = SensorCompressionType.None;
        }

        /// <inheritdoc/>
        protected override int GetCellObservationSize()
        {
            return DetectableTags == null ? 0 : DetectableTags.Length;
        }

        /// <inheritdoc/>
        protected override bool IsDataNormalized()
        {
            return false;
        }

        /// <inheritdoc/>
        protected internal override bool ProcessAllCollidersInCell()
        {
            return true;
        }

        /// <summary>
        /// Get the count of object for each detectable tags detected in a cell.
        /// </summary>
        /// <param name="detectedObject">The game object that was detected within a certain cell</param>
        /// <param name="tagIndex">The index of the detectedObject's tag in the DetectableObjects list</param>
        /// <param name="dataBuffer">The buffer to write the observation values.
        ///         The buffer size is configured by <seealso cref="GetCellObservationSize"/>.
        /// </param>
        protected override void GetObjectData(GameObject currentColliderGo, int typeIndex, float[] dataBuffer)
        {
            dataBuffer[typeIndex] += 1;
        }
    }
}
