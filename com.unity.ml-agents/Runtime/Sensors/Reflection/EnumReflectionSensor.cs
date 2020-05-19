using System;
using UnityEngine;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal class EnumReflectionSensor: ReflectionSensorBase
    {
        Array m_Values;

        internal EnumReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, GetEnumObservationSize(reflectionSensorInfo))
        {
            var t = reflectionSensorInfo.FieldInfo != null ? reflectionSensorInfo.FieldInfo.FieldType : reflectionSensorInfo.PropertyInfo.PropertyType;
            m_Values = Enum.GetValues(t);
        }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var enumValue = GetReflectedValue();
            int i = 0;
            foreach(var val in m_Values)
            {
                //Debug.Log($"{val.GetType()}  {enumValue.GetType()}");
                if (val.Equals(enumValue))
                {
                    writer[i] = 1.0f;
                }
                else
                {
                    writer[i] = 0.0f;
                }

                i++;
            }
        }

        internal static int GetEnumObservationSize(ReflectionSensorInfo reflectionSensorInfo)
        {
            var t = reflectionSensorInfo.FieldInfo != null ? reflectionSensorInfo.FieldInfo.FieldType : reflectionSensorInfo.PropertyInfo.PropertyType;
            return GetEnumObservationSize(t);
        }

        internal static int GetEnumObservationSize(Type t)
        {
            // TODO allow for unknown value
            var values = Enum.GetValues(t);
            return values.Length;
        }
    }
}
