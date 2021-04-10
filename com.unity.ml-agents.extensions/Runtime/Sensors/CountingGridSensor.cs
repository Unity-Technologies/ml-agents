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

        protected override int GetCellObservationSize()
        {
            return DetectableTags.Length;
        }

        protected override bool IsDataNormalized()
        {
            return false;
        }

        protected internal override bool ProcessAllCollidersInCell()
        {
            return true;
        }

        protected override void GetObjectData(GameObject currentColliderGo, int typeIndex, float[] dataBuffer)
        {
            dataBuffer[typeIndex] += 1;
        }
    }
}
