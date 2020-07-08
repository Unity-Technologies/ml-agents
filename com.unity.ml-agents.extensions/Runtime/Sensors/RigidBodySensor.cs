using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// ISensor implementation that generates observations for a group of Rigidbodies and Joints.
    /// </summary>
    public class RigidBodySensor : ISensor
    {
        int[] m_Shape;
        string m_SensorName;

        PoseExtractor m_PoseExtractor;
        PhysicsSensorSettings m_Settings;

        /// <summary>
        ///  Construct a new RigidBodySensor
        /// </summary>
        /// <param name="rootBody"></param>
        /// <param name="settings"></param>
        /// <param name="sensorName"></param>
        public RigidBodySensor(Rigidbody rootBody, PhysicsSensorSettings settings, string sensorName=null)
        {
            m_PoseExtractor = new RigidBodyPoseExtractor(rootBody);
            m_SensorName = string.IsNullOrEmpty(sensorName) ? $"RigidBodySensor:{rootBody?.name}" : sensorName;
            m_Settings = settings;

            var numTransformObservations = settings.TransformSize(m_PoseExtractor.NumPoses);
            m_Shape = new[] { numTransformObservations };
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            var numWritten = writer.WritePoses(m_Settings, m_PoseExtractor);
            return numWritten;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public void Update()
        {
            if (m_Settings.UseModelSpace)
            {
                m_PoseExtractor.UpdateModelSpacePoses();
            }

            if (m_Settings.UseLocalSpace)
            {
                m_PoseExtractor.UpdateLocalSpacePoses();
            }
        }

        /// <inheritdoc/>
        public void Reset() {}

        /// <inheritdoc/>
        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_SensorName;
        }
    }
}
