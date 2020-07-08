using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Editor component that creates a RigidBodySensor for the Agent.
    /// </summary>
    public class RigidBodySensorComponent  : SensorComponent
    {
        /// <summary>
        /// The root Rigidbody of the system.
        /// </summary>
        public Rigidbody RootBody;

        /// <summary>
        /// Settings defining what types of observations will be generated.
        /// </summary>
        [SerializeField]
        public PhysicsSensorSettings Settings = PhysicsSensorSettings.Default();

        /// <summary>
        /// Optional sensor name. This must be unique for each Agent.
        /// </summary>
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
