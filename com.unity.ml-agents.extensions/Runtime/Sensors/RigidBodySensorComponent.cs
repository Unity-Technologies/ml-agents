using System.Collections.Generic;
using System.Net.Configuration;
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
        [SerializeField]
        public string sensorName;

        [SerializeField]
        [HideInInspector]
        RigidBodyPoseExtractor m_PoseExtractor;

        /// <summary>
        /// Creates a PhysicsBodySensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            var _sensorName = string.IsNullOrEmpty(sensorName) ? $"PhysicsBodySensor:{RootBody?.name}" : sensorName;
            return new PhysicsBodySensor(GetPoseExtractor(), Settings, _sensorName);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            if (RootBody == null)
            {
                return new[] { 0 };
            }

            var poseExtractor = GetPoseExtractor();
            var numPoseObservations = poseExtractor.GetNumPoseObservations(Settings);

            var numJointObservations = 0;
            // Start from i=1 to ignore the root
            // TODO ignore joints on disabled bodies
            var numBodies = poseExtractor.Bodies?.Length ?? 0;
            for (var i = 1; i < numBodies; i++)
            {
                var body = poseExtractor.Bodies[i];
                var joint = body?.GetComponent<Joint>();
                numJointObservations += RigidBodyJointExtractor.NumObservations(body, joint, Settings);
            }
            return new[] { numPoseObservations + numJointObservations };
        }

        /// <summary>
        /// Get the DisplayNodes of the hierarchy.
        /// </summary>
        /// <returns></returns>
        internal List<PoseExtractor.DisplayNode> GetDisplayNodes()
        {
            return GetPoseExtractor().GetDisplayNodes();
        }

        /// <summary>
        /// Lazy construction of the PoseExtractor.
        /// </summary>
        /// <returns></returns>
        RigidBodyPoseExtractor GetPoseExtractor()
        {
            if (m_PoseExtractor == null)
            {
                ResetPoseExtractor();
            }

            return m_PoseExtractor;
        }

        /// <summary>
        /// Reset the pose extractor, trying to keep the enabled state of the corresponding poses the same.
        /// </summary>
        internal void ResetPoseExtractor()
        {
            // Get the current enabled state of each body, so that we can reinitialize with them.
            Dictionary<Rigidbody, bool> bodyPosesEnabled = null;
            if (m_PoseExtractor != null)
            {
                bodyPosesEnabled = m_PoseExtractor.GetBodyPosesEnabled();
            }
            m_PoseExtractor = new RigidBodyPoseExtractor(RootBody, gameObject, VirtualRoot, bodyPosesEnabled);
        }

        /// <summary>
        /// Toggle the pose at the given index.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="enabled"></param>
        internal void SetPoseEnabled(int index, bool enabled)
        {
            GetPoseExtractor().SetPoseEnabled(index, enabled);
        }
    }

}
