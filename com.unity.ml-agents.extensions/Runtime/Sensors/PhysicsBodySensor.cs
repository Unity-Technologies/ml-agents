using System.Collections.Generic;
#if UNITY_2020_1_OR_NEWER
using UnityEngine;
#endif
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// ISensor implementation that generates observations for a group of Rigidbodies or ArticulationBodies.
    /// </summary>
    public class PhysicsBodySensor : ISensor
    {
        int[] m_Shape;
        string m_SensorName;

        PoseExtractor m_PoseExtractor;
        List<IJointExtractor> m_JointExtractors;
        PhysicsSensorSettings m_Settings;

        /// <summary>
        /// Construct a new PhysicsBodySensor
        /// </summary>
        /// <param name="poseExtractor"></param>
        /// <param name="settings"></param>
        /// <param name="sensorName"></param>
        public PhysicsBodySensor(
            RigidBodyPoseExtractor poseExtractor,
            PhysicsSensorSettings settings,
            string sensorName
        )
        {
            m_PoseExtractor = poseExtractor;
            m_SensorName = sensorName;
            m_Settings = settings;

            var numJointExtractorObservations = 0;
            m_JointExtractors = new List<IJointExtractor>(poseExtractor.NumEnabledPoses);
            foreach (var rb in poseExtractor.GetEnabledRigidbodies())
            {
                var jointExtractor = new RigidBodyJointExtractor(rb);
                numJointExtractorObservations += jointExtractor.NumObservations(settings);
                m_JointExtractors.Add(jointExtractor);
            }

            var numTransformObservations = m_PoseExtractor.GetNumPoseObservations(settings);
            m_Shape = new[] { numTransformObservations + numJointExtractorObservations };
        }

#if UNITY_2020_1_OR_NEWER
        public PhysicsBodySensor(ArticulationBody rootBody, PhysicsSensorSettings settings, string sensorName=null)
        {
            var poseExtractor = new ArticulationBodyPoseExtractor(rootBody);
            m_PoseExtractor = poseExtractor;
            m_SensorName = string.IsNullOrEmpty(sensorName) ? $"ArticulationBodySensor:{rootBody?.name}" : sensorName;
            m_Settings = settings;

            var numJointExtractorObservations = 0;
            m_JointExtractors = new List<IJointExtractor>(poseExtractor.NumEnabledPoses);
            foreach(var articBody in poseExtractor.GetEnabledArticulationBodies())
            {
                var jointExtractor = new ArticulationBodyJointExtractor(articBody);
                numJointExtractorObservations += jointExtractor.NumObservations(settings);
                m_JointExtractors.Add(jointExtractor);
            }

            var numTransformObservations = m_PoseExtractor.GetNumPoseObservations(settings);
            m_Shape = new[] { numTransformObservations + numJointExtractorObservations };
        }
#endif

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            var numWritten = writer.WritePoses(m_Settings, m_PoseExtractor);
            foreach (var jointExtractor in m_JointExtractors)
            {
                numWritten += jointExtractor.Write(m_Settings, writer, numWritten);
            }
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
        public void Reset() { }

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
