using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Specify that a field or property should be used to generate observations for an Agent.
    /// For each field or property that uses ObservableAttribute, a corresponding
    /// <see cref="ISensor"/> will be created during Agent initialization, and this
    /// sensor will read the values during training and inference.
    /// </summary>
    /// <remarks>
    /// ObservableAttribute is intended to make initial setup of an Agent easier. Because it
    /// uses reflection to read the values of fields and properties at runtime, this may
    /// be much slower than reading the values directly. If the performance of
    /// ObservableAttribute is an issue, you can get the same functionality by overriding
    /// <see cref="Agent.CollectObservations(VectorSensor)"/> or creating a custom
    /// <see cref="ISensor"/> implementation to read the values without reflection.
    ///
    /// Note that you do not need to adjust the VectorObservationSize in
    /// <see cref="Unity.MLAgents.Policies.BrainParameters"/> when adding ObservableAttribute
    /// to fields or properties.
    /// </remarks>
    /// <example>
    /// This sample class will produce two observations, one for the m_Health field, and one
    /// for the HealthPercent property.
    /// <code>
    /// using Unity.MLAgents;
    /// using Unity.MLAgents.Sensors.Reflection;
    ///
    /// public class MyAgent : Agent
    /// {
    ///     [Observable]
    ///     int m_Health;
    ///
    ///     [Observable]
    ///     float HealthPercent
    ///     {
    ///         get => return 100.0f * m_Health / float(m_MaxHealth);
    ///     }
    /// }
    /// </code>
    /// </example>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class ObservableAttribute : Attribute
    {
        string m_Name;
        int m_NumStackedObservations;

        /// <summary>
        /// Default binding flags used for reflection of members and properties.
        /// </summary>
        const BindingFlags k_BindingFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        /// <summary>
        /// Supported types and their observation sizes and corresponding sensor type.
        /// </summary>
        static Dictionary<Type, (int, Type)> s_TypeToSensorInfo = new Dictionary<Type, (int, Type)>()
        {
            {typeof(int), (1, typeof(IntReflectionSensor))},
            {typeof(bool), (1, typeof(BoolReflectionSensor))},
            {typeof(float), (1, typeof(FloatReflectionSensor))},

            {typeof(Vector2), (2, typeof(Vector2ReflectionSensor))},
            {typeof(Vector3), (3, typeof(Vector3ReflectionSensor))},
            {typeof(Vector4), (4, typeof(Vector4ReflectionSensor))},
            {typeof(Quaternion), (4, typeof(QuaternionReflectionSensor))},
        };

        /// <summary>
        /// ObservableAttribute constructor.
        /// </summary>
        /// <param name="name">Optional override for the sensor name. Note that all sensors for an Agent
        /// must have a unique name.</param>
        /// <param name="numStackedObservations">Number of frames to concatenate observations from.</param>
        public ObservableAttribute(string name = null, int numStackedObservations = 1)
        {
            m_Name = name;
            m_NumStackedObservations = numStackedObservations;
        }

        /// <summary>
        /// Returns a FieldInfo for all fields that have an ObservableAttribute
        /// </summary>
        /// <param name="o">Object being reflected</param>
        /// <param name="excludeInherited">Whether to exclude inherited properties or not.</param>
        /// <returns></returns>
        static IEnumerable<(FieldInfo, ObservableAttribute)> GetObservableFields(object o, bool excludeInherited)
        {
            // TODO cache these (and properties) by type, so that we only have to reflect once.
            var bindingFlags = k_BindingFlags | (excludeInherited ? BindingFlags.DeclaredOnly : 0);
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

        /// <summary>
        /// Returns a PropertyInfo for all fields that have an ObservableAttribute
        /// </summary>
        /// <param name="o">Object being reflected</param>
        /// <param name="excludeInherited">Whether to exclude inherited properties or not.</param>
        /// <returns></returns>
        static IEnumerable<(PropertyInfo, ObservableAttribute)> GetObservableProperties(object o, bool excludeInherited)
        {
            var bindingFlags = k_BindingFlags | (excludeInherited ? BindingFlags.DeclaredOnly : 0);
            var properties = o.GetType().GetProperties(bindingFlags);
            foreach (var prop in properties)
            {
                var attr = (ObservableAttribute)GetCustomAttribute(prop, typeof(ObservableAttribute));
                if (attr != null)
                {
                    yield return (prop, attr);
                }
            }
        }

        /// <summary>
        /// Creates sensors for each field and property with ObservableAttribute.
        /// </summary>
        /// <param name="o">Object being reflected</param>
        /// <param name="excludeInherited">Whether to exclude inherited properties or not.</param>
        /// <returns></returns>
        internal static List<ISensor> CreateObservableSensors(object o, bool excludeInherited)
        {
            var sensorsOut = new List<ISensor>();
            foreach (var(field, attr) in GetObservableFields(o, excludeInherited))
            {
                var sensor = CreateReflectionSensor(o, field, null, attr);
                if (sensor != null)
                {
                    sensorsOut.Add(sensor);
                }
            }

            foreach (var(prop, attr) in GetObservableProperties(o, excludeInherited))
            {
                if (!prop.CanRead)
                {
                    // Skip unreadable properties.
                    continue;
                }
                var sensor = CreateReflectionSensor(o, null, prop, attr);
                if (sensor != null)
                {
                    sensorsOut.Add(sensor);
                }
            }

            return sensorsOut;
        }

        /// <summary>
        /// Create the ISensor for either the field or property on the provided object.
        /// If the data type is unsupported, or the property is write-only, returns null.
        /// </summary>
        /// <param name="o"></param>
        /// <param name="fieldInfo"></param>
        /// <param name="propertyInfo"></param>
        /// <param name="observableAttribute"></param>
        /// <returns></returns>
        /// <exception cref="UnityAgentsException"></exception>
        static ISensor CreateReflectionSensor(object o, FieldInfo fieldInfo, PropertyInfo propertyInfo, ObservableAttribute observableAttribute)
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

            if (!s_TypeToSensorInfo.ContainsKey(memberType) && !memberType.IsEnum)
            {
                // For unsupported types, return null and we'll filter them out later.
                return null;
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
            if (memberType.IsEnum)
            {
                sensor = new EnumReflectionSensor(reflectionSensorInfo);
            }
            else
            {
                var (_, sensorType) = s_TypeToSensorInfo[memberType];
                sensor = (ISensor) Activator.CreateInstance(sensorType, reflectionSensorInfo);
            }

            // Wrap the base sensor in a StackingSensor if we're using stacking.
            if (observableAttribute.m_NumStackedObservations > 1)
            {
                return new StackingSensor(sensor, observableAttribute.m_NumStackedObservations);
            }

            return sensor;
        }

        /// <summary>
        /// Gets the sum of the observation sizes of the Observable fields and properties on an object.
        /// Also appends errors to the errorsOut array.
        /// </summary>
        /// <param name="o"></param>
        /// <param name="excludeInherited"></param>
        /// <param name="errorsOut"></param>
        /// <returns></returns>
        internal static int GetTotalObservationSize(object o, bool excludeInherited, List<string> errorsOut)
        {
            int sizeOut = 0;
            foreach (var(field, attr) in GetObservableFields(o, excludeInherited))
            {
                if (s_TypeToSensorInfo.ContainsKey(field.FieldType))
                {
                    var (obsSize, _) = s_TypeToSensorInfo[field.FieldType];
                    sizeOut += obsSize * attr.m_NumStackedObservations;
                }
                else if (field.FieldType.IsEnum)
                {
                    sizeOut += EnumReflectionSensor.GetEnumObservationSize(field.FieldType);
                }
                else
                {
                    errorsOut.Add($"Unsupported Observable type {field.FieldType.Name} on field {field.Name}");
                }
            }

            foreach (var(prop, attr) in GetObservableProperties(o, excludeInherited))
            {
                if (!prop.CanRead)
                {
                    errorsOut.Add($"Observable property {prop.Name} is write-only.");
                }
                else if (s_TypeToSensorInfo.ContainsKey(prop.PropertyType))
                {
                    var (obsSize, _) = s_TypeToSensorInfo[prop.PropertyType];
                    sizeOut += obsSize * attr.m_NumStackedObservations;
                }
                else if (prop.PropertyType.IsEnum)
                {
                    sizeOut += EnumReflectionSensor.GetEnumObservationSize(prop.PropertyType);
                }
                else
                {
                    errorsOut.Add($"Unsupported Observable type {prop.PropertyType.Name} on property {prop.Name}");
                }
            }

            return sizeOut;
        }
    }
}
