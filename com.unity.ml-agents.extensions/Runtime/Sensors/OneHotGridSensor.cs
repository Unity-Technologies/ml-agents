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

        protected override int GetCellObservationSize()
        {
            return DetectableTags.Length;
        }

        protected override bool IsDataNormalized()
        {
            return true;
        }

        protected internal override bool ProcessAllCollidersInCell()
        {
            return false;
        }

        protected override void GetObjectData(GameObject currentColliderGo, int typeIndex, float[] dataBuffer)
        {
            dataBuffer[typeIndex] = 1;
        }
    }
}
