using System.Reflection;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal class AttributePropertySensor : ISensor
    {
        object m_Object;
        PropertyInfo m_PropertyInfo;
        // Not currently used, but might want later.
        ObservableAttribute m_ObservableAttribute;

        string m_SensorName;
        int[] m_Shape;

        public AttributePropertySensor(object o, PropertyInfo propertyInfo, ObservableAttribute observableAttribute)
        {
            m_Object = o;
            m_PropertyInfo = propertyInfo;
            m_ObservableAttribute = observableAttribute;

            m_SensorName = $"ObservableAttribute:{propertyInfo.DeclaringType.Name}.{propertyInfo.Name}";
            // TODO handle scalar, quaternion, blittable(?)
            m_Shape = new [] {3};
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            // TODO make Delegate in ctor instead?
            var val = m_PropertyInfo.GetMethod.Invoke(m_Object, null);

            if (m_PropertyInfo.PropertyType == typeof(UnityEngine.Vector3))
            {
                var vec3Val = (UnityEngine.Vector3)val;
                writer[0] = vec3Val.x;
                writer[1] = vec3Val.y;
                writer[2] = vec3Val.z;
            }
            else
            {
                writer[0] = 0.0f;
                writer[1] = 0.0f;
                writer[2] = 0.0f;
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
