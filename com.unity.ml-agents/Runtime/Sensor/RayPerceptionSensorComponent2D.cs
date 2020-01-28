using UnityEngine;

namespace MLAgents.Sensor
{
    [AddComponentMenu("ML Agents/Ray Perception Sensor 2D", (int) MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent2D : RayPerceptionSensorComponentBase
    {
        public override RayPerceptionSensor.CastType GetCastType()
        {
            return RayPerceptionSensor.CastType.Cast2D;
        }
    }
}
