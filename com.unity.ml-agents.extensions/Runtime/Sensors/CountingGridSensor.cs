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
            int[] channelDepths,
            string[] detectableObjects,
            SensorCompressionType compression
        ) : base(name, cellScale, gridNum, channelDepths, detectableObjects, compression)
        {
            CompressionType = SensorCompressionType.None;
        }

        protected override void GetObjectData(GameObject currentColliderGo, int typeIndex, float[] dataBuffer)
        {
            dataBuffer[typeIndex] += 1;
        }
    }
}
