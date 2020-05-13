using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace Unity.MLAgents.Sensors.Reflection
{
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class ObservableAttribute : Attribute
    {
        string m_Name;
        int m_NumStackedObservations;

        const BindingFlags k_BindingFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        static Dictionary<Type, int> s_TypeSizes = new Dictionary<Type, int>()
        {
            {typeof(int), 1},
            {typeof(bool), 1},
            {typeof(float), 1},
            {typeof(Vector2), 2},
            {typeof(Vector3), 3},
            {typeof(Vector4), 4},
            {typeof(Quaternion), 4},
        };

        /// <summary>
        /// ObservableAttribute constructor.
        /// </summary>
        /// <param name="name">Optional override for the sensor name. Note that all sensors for an Agent
        /// must have a unique name.</param>
        /// <param name="numStackedObservations">Number of frames to concatenate observations from.</param>
        public ObservableAttribute(string name=null, int numStackedObservations=1)
        {
            m_Name = name;
            m_NumStackedObservations = numStackedObservations;
        }

        internal static IEnumerable<(FieldInfo, ObservableAttribute)> GetObservableFields(object o, bool declaredOnly)
        {
            // TODO cache these (and properties) by type, so that we only have to reflect once.
            var bindingFlags = k_BindingFlags | (declaredOnly ? BindingFlags.DeclaredOnly : 0);
            var fields = o.GetType().GetFields(bindingFlags);
            foreach (var field in fields)
            {
                var attr = (ObservableAttribute)GetCustomAttribute(field, typeof(ObservableAttribute));
                if (attr != null)
                {
                    yield return (field, attr);
                }
            }
        }

        internal static IEnumerable<(PropertyInfo, ObservableAttribute)> GetObservableProperties(object o, bool declaredOnly)
        {
            var bindingFlags = k_BindingFlags | (declaredOnly ? BindingFlags.DeclaredOnly : 0);
            var properties = o.GetType().GetProperties(bindingFlags);
            foreach (var prop in properties)
            {
                if (!prop.CanRead)
                {
                    // Ignore write-only properties.
                    continue;
                }
                var attr = (ObservableAttribute)GetCustomAttribute(prop, typeof(ObservableAttribute));
                if (attr != null)
                {
                    yield return (prop, attr);
                }
            }
        }

        internal static List<ISensor> GetObservableSensors(object o, bool declaredOnly)
        {
            var sensorsOut = new List<ISensor>();
            foreach (var (field, attr) in GetObservableFields(o, declaredOnly))
            {
                sensorsOut.Add(CreateReflectionSensor(o, field, null, attr));
            }

            foreach (var (prop, attr) in GetObservableProperties(o, declaredOnly))
            {
                sensorsOut.Add(CreateReflectionSensor(o, null, prop, attr));
            }

            return sensorsOut;
        }

        internal static ISensor CreateReflectionSensor(object o, FieldInfo fieldInfo, PropertyInfo propertyInfo, ObservableAttribute observableAttribute)
        {
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

            var reflectionSensorInfo = new ReflectionSensorInfo
            {
                Object = o,
                FieldInfo = fieldInfo,
                PropertyInfo = propertyInfo,
                ObservableAttribute = observableAttribute,
                SensorName = sensorName
            };

            ISensor sensor = null;
            if (memberType == typeof(Int32))
            {
                sensor = new IntReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(float))
            {
                sensor = new  FloatReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(bool))
            {
                sensor = new BoolReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(Vector2))
            {
                sensor = new Vector2ReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(Vector3))
            {
                sensor = new Vector3ReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(Vector4))
            {
                sensor = new Vector4ReflectionSensor(reflectionSensorInfo);
            }
            if (memberType == typeof(Quaternion))
            {
                sensor = new QuaternionReflectionSensor(reflectionSensorInfo);
            }

            if (sensor == null)
            {
                throw new UnityAgentsException($"Unsupported Observable type: {memberType.Name}");
            }

            // Wrap the base sensor in a StackingSensor if we're using stacking.
            if (observableAttribute.m_NumStackedObservations > 1)
            {
                return new StackingSensor(sensor, observableAttribute.m_NumStackedObservations);
            }

            return sensor;
        }

        internal static int GetTotalObservationSize(object o, bool declaredOnly, List<string> errorsOut)
        {
            int sizeOut = 0;
            foreach (var (field, attr) in GetObservableFields(o, declaredOnly))
            {
                if (s_TypeSizes.ContainsKey(field.FieldType))
                {
                    sizeOut += s_TypeSizes[field.FieldType] * attr.m_NumStackedObservations;
                }
                else
                {
                    errorsOut.Add($"Unsupported Observable type {field.FieldType.Name} on field {field.Name}");
                }
            }

            foreach (var (prop, attr) in GetObservableProperties(o, declaredOnly))
            {

                if (s_TypeSizes.ContainsKey(prop.PropertyType))
                {
                    sizeOut += s_TypeSizes[prop.PropertyType] * attr.m_NumStackedObservations;
                }
                else
                {
                    errorsOut.Add($"Unsupported Observable type {prop.PropertyType.Name} on field {prop.Name}");
                }
            }

            return sizeOut;
        }

    }

}
