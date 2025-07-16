#if UNITY_2020_1_OR_NEWER
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class ArticulationBodySensorComponent : SensorComponent
    {
        public ArticulationBody RootBody;

        [SerializeField]
        public PhysicsSensorSettings Settings = PhysicsSensorSettings.Default();
        public string sensorName;

        /// <summary>
        /// Creates a PhysicsBodySensor.
        /// </summary>
        /// <returns>Corresponding sensors.</returns>
        public override ISensor[] CreateSensors()
        {
            return new ISensor[] {new PhysicsBodySensor(RootBody, Settings, sensorName)};
        }
    }
}
#endif // UNITY_2020_1_OR_NEWER
