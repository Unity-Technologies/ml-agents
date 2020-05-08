using System;
using System.Collections.Generic;
using System.Reflection;

namespace Unity.MLAgents.Sensors.Reflection
{
    [System.AttributeUsage(System.AttributeTargets.Field | System.AttributeTargets.Property)]
    public class ObservableAttribute : System.Attribute
    {
        // Currently nothing here
        string m_Name;

        public ObservableAttribute(string name=null)
        {
            m_Name = name;
        }

        internal static List<ISensor> GetObservableSensors(object o)
        {
            var sensorsOut = new List<ISensor>();

            var fields = o.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            foreach (var field in fields)
            {
                var attr = (ObservableAttribute)Attribute.GetCustomAttribute(field, typeof(ObservableAttribute));
                if (attr != null)
                {
                    sensorsOut.Add(CreateReflectionSensor(o, field, null, attr));
                }
            }

            var properties = o.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            foreach (var prop in properties)
            {
                var attr = (ObservableAttribute)Attribute.GetCustomAttribute(prop, typeof(ObservableAttribute));
                if (attr != null)
                {
                    sensorsOut.Add(CreateReflectionSensor(o, null, prop, attr));
                }
            }
            return sensorsOut;
        }

        internal static ISensor CreateReflectionSensor(object o, FieldInfo fieldInfo, PropertyInfo propertyInfo, ObservableAttribute observableAttribute)
        {
            MemberInfo memberInfo = fieldInfo != null ? (MemberInfo) fieldInfo : propertyInfo;
            string memberName;
            string declaringTypeName;
            Type memberType;
            if (fieldInfo != null)
            {
                declaringTypeName = fieldInfo.DeclaringType.Name;
                memberName = fieldInfo.Name;
                memberType = fieldInfo.FieldType;
            }
            else
            {
                declaringTypeName = propertyInfo.DeclaringType.Name;
                memberName = propertyInfo.Name;
                memberType = propertyInfo.PropertyType;
            }

            string sensorName;
            if (string.IsNullOrEmpty(observableAttribute.m_Name))
            {
                sensorName = $"ObservableAttribute:{declaringTypeName}.{memberName}";
            }
            else
            {
                sensorName = observableAttribute.m_Name;
            }

            if (memberType == typeof(System.Int32))
            {
                return new IntReflectionSensor(o, fieldInfo, propertyInfo, observableAttribute, sensorName);
            }

            throw new UnityAgentsException($"Unsupported Observable type: {memberType.Name}");

        }

    }

}
