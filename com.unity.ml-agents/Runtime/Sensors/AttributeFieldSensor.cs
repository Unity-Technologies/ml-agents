using System.Reflection;

namespace Unity.MLAgents.Sensors
{
    internal class AttributeFieldSensor : ISensor
    {
        object m_Object;
        FieldInfo m_FieldInfo;
        // Not currently used, but might want later.
        ObservableAttribute m_ObservableAttribute;

        string m_SensorName;
        int[] m_Shape;

        public AttributeFieldSensor(object o, FieldInfo fieldInfo, ObservableAttribute observableAttribute)
        {
            m_Object = o;
            m_FieldInfo = fieldInfo;
            m_ObservableAttribute = observableAttribute;

            m_SensorName = $"ObservableAttribute:{fieldInfo.DeclaringType.Name}.{fieldInfo.Name}";
            // TODO handle Vector3, quaternion, blittable(?)
            m_Shape = new [] {1};
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            var val = m_FieldInfo.GetValue(m_Object);
            if (m_FieldInfo.FieldType == typeof(System.Boolean))
            {
                var boolVal = (System.Boolean)val;
                writer[0] = boolVal ? 1.0f : 0.0f;
            }
            else
            {
                writer[0] = 0.0f;
            }
            return 1;
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
