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
        /// Optional GameObject used to determine the root of the poses.
        /// </summary>
        public GameObject VirtualRoot;

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
            return new PhysicsBodySensor(RootBody, gameObject, VirtualRoot, Settings, sensorName);
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
            var poseExtractor = new RigidBodyPoseExtractor(RootBody, gameObject, VirtualRoot);
            var numPoseObservations = poseExtractor.GetNumPoseObservations(Settings);

            var numJointObservations = 0;
            // Start from i=1 to ignore the root
            for (var i = 1; i < poseExtractor.Bodies.Length; i++)
            {
                var body = poseExtractor.Bodies[i];
                var joint = body?.GetComponent<Joint>();
                numJointObservations += RigidBodyJointExtractor.NumObservations(body, joint, Settings);
            }
            return new[] { numPoseObservations + numJointObservations };
        }
    }

}
