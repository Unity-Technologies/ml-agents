using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
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
