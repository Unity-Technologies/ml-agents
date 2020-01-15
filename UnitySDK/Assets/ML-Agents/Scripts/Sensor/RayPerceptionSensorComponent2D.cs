using UnityEngine;

namespace MLAgents.Sensor
{
    [AddComponentMenu("ML Agents/Sensors/Ray Perception Sensor 2D")]
    public class RayPerceptionSensorComponent2D : RayPerceptionSensorComponentBase
    {
        public override RayPerceptionSensor.CastType GetCastType()
        {
            return RayPerceptionSensor.CastType.Cast2D;
        }
    }
}
