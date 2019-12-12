namespace MLAgents.Sensor
{
    public class RayPerceptionSensorComponent2D : RayPerceptionSensorComponentBase
    {
        public override RayPerceptionSensor.CastType GetCastType()
        {
            return RayPerceptionSensor.CastType.Cast2D;
        }
    }
}
