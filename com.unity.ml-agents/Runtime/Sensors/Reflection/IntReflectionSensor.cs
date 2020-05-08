using System.Reflection;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal class IntReflectionSensor : ReflectionSensorBase
    {
        internal IntReflectionSensor(object o, FieldInfo fieldInfo, PropertyInfo propertyInfo, ObservableAttribute observableAttribute, string sensorName)
            : base(o, fieldInfo, propertyInfo, observableAttribute, 1, sensorName)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            if (m_FieldInfo != null)
            {
                var val = m_FieldInfo.GetValue(m_Object);
                var intVal = (System.Int32)val;
                writer[0] = intVal;
            }
            else
            {
                // TODO form delegate in ctor
                var val = m_PropertyInfo.GetMethod.Invoke(m_Object, null);
                var intVal = (System.Int32)val;
                writer[0] = intVal;
            }
        }
    }
}
