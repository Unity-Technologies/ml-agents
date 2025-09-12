using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Extracts joint and rigidbody information for physics-based sensors.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.MLAgents.Extensions.Sensors")]
    public class RigidBodyJointExtractor : IJointExtractor
    {
        Rigidbody m_Body;
        Joint m_Joint;

        /// <summary>
        /// Initializes a new instance of the <see cref="RigidBodyJointExtractor"/> class.
        /// </summary>
        /// <param name="body">The Rigidbody to extract joint information from.</param>
        public RigidBodyJointExtractor(Rigidbody body)
        {
            m_Body = body;
            m_Joint = m_Body?.GetComponent<Joint>();
        }

        /// <summary>
        /// Gets the number of observations for this joint extractor using the provided settings.
        /// </summary>
        /// <param name="settings">The physics sensor settings.</param>
        /// <returns>The number of observations for this joint extractor.</returns>
        public int NumObservations(PhysicsSensorSettings settings)
        {
            return NumObservations(m_Body, m_Joint, settings);
        }

        /// <summary>
        /// Gets the number of observations for the specified rigidbody and joint using the provided settings.
        /// </summary>
        /// <param name="body">The Rigidbody to extract from.</param>
        /// <param name="joint">The Joint to extract from.</param>
        /// <param name="settings">The physics sensor settings.</param>
        /// <returns>The number of observations for the specified rigidbody and joint.</returns>
        public static int NumObservations(Rigidbody body, Joint joint, PhysicsSensorSettings settings)
        {
            if (body == null || joint == null)
            {
                return 0;
            }

            var numObservations = 0;
            if (settings.UseJointForces)
            {
                // 3 force and 3 torque values
                numObservations += 6;
            }

            return numObservations;
        }

        /// <summary>
        /// Writes the joint observations to the provided writer using the given settings.
        /// </summary>
        /// <param name="settings">The physics sensor settings.</param>
        /// <param name="writer">The observation writer.</param>
        /// <param name="offset">The offset in the writer to start writing at.</param>
        /// <returns>The number of floats written to the writer.</returns>
        public int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset)
        {
            if (m_Body == null || m_Joint == null)
            {
                return 0;
            }

            var currentOffset = offset;
            if (settings.UseJointForces)
            {
                // Take tanh of the forces and torques to ensure they're in [-1, 1]
                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentForce.x);
                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentForce.y);
                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentForce.z);

                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentTorque.x);
                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentTorque.y);
                writer[currentOffset++] = (float)System.Math.Tanh(m_Joint.currentTorque.z);
            }
            return currentOffset - offset;
        }
    }
}
