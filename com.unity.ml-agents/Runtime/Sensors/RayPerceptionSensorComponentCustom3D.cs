using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    public class CustomRayPerceptionOutput : RayPerceptionOutput
    {
        public CustomRayPerceptionOutput()
        {
            CustomObservationSizePerRay = 3;
        }

        public override void GetCustomObservationData(RayOutput rayOutput, float[] buffer)
        {
            var rb = rayOutput.HitGameObject.GetComponent<Rigidbody>();
            if (rb != null)
            {
                buffer[0] = rb.velocity.x;
                buffer[1] = rb.velocity.z;
                buffer[2] = rb.mass;
            }
        }
    }

    /// <summary>
    /// A component for Custom 3D Ray Perception.
    /// </summary>
    [AddComponentMenu("ML Agents/Custom Ray Perception Sensor 3D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponentCustom3D : RayPerceptionSensorComponent3D
    {
        public override ISensor[] CreateSensors()
        {
            var rayPerceptionInput = GetRayPerceptionInput();
            var rayPerceptionOutput = new CustomRayPerceptionOutput();

            m_RaySensor = new RayPerceptionSensor(SensorName, rayPerceptionInput, rayPerceptionOutput);

            if (ObservationStacks != 1)
            {
                var stackingSensor = new StackingSensor(m_RaySensor, ObservationStacks);
                return new ISensor[] { stackingSensor };
            }

            return new ISensor[] { m_RaySensor };
        }
    }
}
