using System.Reflection;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal abstract class ReflectionSensorBase : ISensor
    {
        protected object m_Object;

        protected FieldInfo m_FieldInfo;
        protected PropertyInfo m_PropertyInfo;
        // Not currently used, but might want later.
        protected ObservableAttribute m_ObservableAttribute;

        string m_SensorName;
        int[] m_Shape;

        public ReflectionSensorBase(object o, FieldInfo fieldInfo, PropertyInfo propertyInfo, ObservableAttribute observableAttribute, int size, string sensorName)
        {
            // TODO 2 constructors?

            m_Object = o;
            m_FieldInfo = fieldInfo;
            m_PropertyInfo = propertyInfo;
            m_ObservableAttribute = observableAttribute;
            m_SensorName = sensorName;
            m_Shape = new [] {size};
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            WriteReflectedField(writer);
            return m_Shape[0];
        }

        internal abstract void WriteReflectedField(ObservationWriter writer);

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
