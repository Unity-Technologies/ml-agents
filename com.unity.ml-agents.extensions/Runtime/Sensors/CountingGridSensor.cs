using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor that counts the number of detctable objects.
    /// </summary>
    public class CountingGridSensor : GridSensorBase
    {
        /// <summary>
        /// Create a CountingGridSensor with the specified configuration.
        /// </summary>
        /// <param name="name">The sensor name</param>
        /// <param name="cellScale">The scale of each cell in the grid</param>
        /// <param name="gridSize">Number of cells on each side of the grid</param>
        /// <param name="detectableTags">Tags to be detected by the sensor</param>
        /// <param name="compression">Compression type</param>
        public CountingGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridSize,
            string[] detectableTags,
            SensorCompressionType compression
        ) : base(name, cellScale, gridSize, detectableTags, compression)
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
        protected internal override ProcessCollidersMethod GetProcessCollidersMethod()
        {
            return ProcessCollidersMethod.ProcessAllColliders;
        }

        /// <summary>
        /// Get object counts for each detectable tags detected in a cell.
        /// </summary>
        /// <param name="detectedObject">The game object that was detected within a certain cell</param>
        /// <param name="tagIndex">The index of the detectedObject's tag in the DetectableObjects list</param>
        /// <param name="dataBuffer">The buffer to write the observation values.
        ///         The buffer size is configured by <seealso cref="GetCellObservationSize"/>.
        /// </param>
        protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
        {
            dataBuffer[tagIndex] += 1;
        }
    }
}
