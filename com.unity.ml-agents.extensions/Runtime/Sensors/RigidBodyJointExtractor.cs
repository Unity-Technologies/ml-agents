using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class RigidBodyJointExtractor : IJointExtractor
    {
        Rigidbody m_Body;
        Joint m_Joint;

        public RigidBodyJointExtractor(Rigidbody body)
        {
            m_Body = body;
            m_Joint = m_Body?.GetComponent<Joint>();
        }

        public int NumObservations(PhysicsSensorSettings settings)
        {
            return NumObservations(m_Body, m_Joint, settings);
        }

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
