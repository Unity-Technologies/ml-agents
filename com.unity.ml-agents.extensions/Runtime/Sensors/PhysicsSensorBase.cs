using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public struct PhysicsSensorSettings
    {
        public bool UseModelSpaceTranslations;
        public bool UseModelSpaceRotations;
        public bool UseLocalSpaceTranslations;
        public bool UseLocalSpaceRotations;

        public static PhysicsSensorSettings Default()
        {
            return new PhysicsSensorSettings
            {
                UseModelSpaceTranslations = true,
                UseModelSpaceRotations = true,
            };
        }
    }

    public class PhysicsSensorBase : ISensor
    {
        int[] m_Shape;
        string m_SensorName;

        HierarchyUtil m_HierarchyUtil;
        PhysicsSensorSettings m_Settings;

        public PhysicsSensorBase(HierarchyUtil hierarchyUtil, PhysicsSensorSettings settings)
        {
            m_HierarchyUtil = hierarchyUtil;
            m_Settings = settings;

        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            return 0;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public void Update() {}

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
