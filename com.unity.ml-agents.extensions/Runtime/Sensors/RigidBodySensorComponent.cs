using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class RigidBodySensorComponent  : SensorComponent
    {
        public Rigidbody RootBody;

        [SerializeField]
        public PhysicsSensorSettings Settings = PhysicsSensorSettings.Default();
        public string sensorName;

        /// <summary>
        /// Creates a RigidBodySensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            return new RigidBodySensor(RootBody, Settings, sensorName);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            if (RootBody == null)
            {
                return new[] { 0 };
            }

            // TODO static method in RigidBodySensor?
            // TODO only update PoseExtractor when body changes?
            var poseExtractor = new RigidBodyPoseExtractor(RootBody);
            var numTransformObservations = Settings.TransformSize(poseExtractor.NumPoses);
            return new[] { numTransformObservations };
        }
    }

}
