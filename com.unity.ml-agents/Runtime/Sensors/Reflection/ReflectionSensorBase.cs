using System;
using System.Reflection;

namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Construction info for a ReflectionSensorBase.
    /// </summary>
    internal struct ReflectionSensorInfo
    {
        public object Object;

        public FieldInfo FieldInfo;
        public PropertyInfo PropertyInfo;
        public ObservableAttribute ObservableAttribute;
        public string SensorName;

        public Type GetMemberType()
        {
            return FieldInfo != null ? FieldInfo.FieldType : PropertyInfo.PropertyType;
        }
    }

    /// <summary>
    /// Abstract base class for reflection-based sensors.
    /// </summary>
    internal abstract class ReflectionSensorBase : ISensor, IBuiltInSensor
    {
        protected object m_Object;

        // Exactly one of m_FieldInfo and m_PropertyInfo should be non-null.
        protected FieldInfo m_FieldInfo;
        protected PropertyInfo m_PropertyInfo;

        // Not currently used, but might want later.
        protected ObservableAttribute m_ObservableAttribute;

        // Cached sensor names and shapes.
        string m_SensorName;
        ObservationSpec m_ObservationSpec;
        int m_NumFloats;

        public ReflectionSensorBase(ReflectionSensorInfo reflectionSensorInfo, int size)
        {
            m_Object = reflectionSensorInfo.Object;
            m_FieldInfo = reflectionSensorInfo.FieldInfo;
            m_PropertyInfo = reflectionSensorInfo.PropertyInfo;
            m_ObservableAttribute = reflectionSensorInfo.ObservableAttribute;
            m_SensorName = reflectionSensorInfo.SensorName;
            m_ObservationSpec = ObservationSpec.Vector(size);
            m_NumFloats = size;
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            WriteReflectedField(writer);
            return m_NumFloats;
        }

        internal abstract void WriteReflectedField(ObservationWriter writer);

        /// <summary>
        /// Get either the reflected field, or return the reflected property.
        /// This should be used by implementations in their WriteReflectedField() method.
        /// </summary>
        /// <returns></returns>
        protected object GetReflectedValue()
        {
            return m_FieldInfo != null ?
                m_FieldInfo.GetValue(m_Object) :
                m_PropertyInfo.GetMethod.Invoke(m_Object, null);
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public void Update() { }

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_SensorName;
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.ReflectionSensor;
        }
    }
}
