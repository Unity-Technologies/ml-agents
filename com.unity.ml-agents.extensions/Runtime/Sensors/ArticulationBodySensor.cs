#if UNITY_2020_1_OR_NEWER
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class ArticulationBodySensor : ISensor
    {
        int[] m_Shape;
        string m_SensorName;

        HierarchyUtil m_HierarchyUtil;
        PhysicsSensorSettings m_Settings;

        public ArticulationBodySensor(ArticulationBody rootBody, PhysicsSensorSettings settings, string sensorName=null)
        {
            m_HierarchyUtil = new ArticulationHierarchyUtil(rootBody);
            m_SensorName = string.IsNullOrEmpty(sensorName) ? $"ArticulationBodySensor:{rootBody.name}" : sensorName;
            m_Settings = settings;

            var numTransformObservations = settings.TransformSize(m_HierarchyUtil.NumTransforms);
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
            var numWritten = writer.WriteHierarchy(m_Settings, m_HierarchyUtil);
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
                m_HierarchyUtil.UpdateModelSpaceTransforms();
            }

            if (m_Settings.UseLocalSpace)
            {
                m_HierarchyUtil.UpdateLocalSpaceTransforms();
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
#endif // UNITY_2020_1_OR_NEWER