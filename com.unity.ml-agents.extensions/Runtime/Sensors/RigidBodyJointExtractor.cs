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
            if(body == null || joint == null)
            {
                return 0;
            }


        }

        public int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset)
        {
            return 0;
        }
    }
}
