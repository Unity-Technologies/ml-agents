using UnityEngine.Serialization;

namespace MLAgents.Sensor
{
    [FormerlySerializedAs("RayPerceptionSensorComponent")]
    public class RayPerceptionSensorComponent3D : RayPerceptionSensorComponentBase
    {
        public override RayPerceptionSensor.CastType GetCastType()
        {
            return RayPerceptionSensor.CastType.Cast3D;
        }
    }
}
