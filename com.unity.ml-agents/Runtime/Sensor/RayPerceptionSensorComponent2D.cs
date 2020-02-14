using UnityEngine;

namespace MLAgents
{
    [AddComponentMenu("ML Agents/Ray Perception Sensor 2D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent2D : RayPerceptionSensorComponentBase
    {
        public RayPerceptionSensorComponent2D()
        {
            // Set to the 2D defaults (just in case they ever diverge).
            rayLayerMask = Physics2D.DefaultRaycastLayers;
        }

        public override RayPerceptionSensor.CastType GetCastType()
        {
            return RayPerceptionSensor.CastType.Cast2D;
        }
    }
}
