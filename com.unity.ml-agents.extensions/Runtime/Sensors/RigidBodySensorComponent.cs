using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Editor component that creates a PhysicsBodySensor for the Agent.
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
        /// Creates a PhysicsBodySensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            return new PhysicsBodySensor(RootBody, gameObject, Settings, sensorName);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            if (RootBody == null)
            {
                return new[] { 0 };
            }

            // TODO static method in PhysicsBodySensor?
            // TODO only update PoseExtractor when body changes?
            var poseExtractor = new RigidBodyPoseExtractor(RootBody, gameObject);
            var numTransformObservations = Settings.TransformSize(poseExtractor.NumPoses);
            return new[] { numTransformObservations };
        }
    }

}
