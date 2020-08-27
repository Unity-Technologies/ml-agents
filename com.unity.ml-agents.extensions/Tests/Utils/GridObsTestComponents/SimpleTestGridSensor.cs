using UnityEngine;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.TestUtils.Sensors
{
    public class SimpleTestGridSensor : GridSensor
    {
        protected override float[] GetObjectData(GameObject currentColliderGo,
            float type_index, float normalized_distance)
        {
            return (float[])currentColliderGo.GetComponent<GridSensorDummyData>().Data.Clone();
        }
    }
}
