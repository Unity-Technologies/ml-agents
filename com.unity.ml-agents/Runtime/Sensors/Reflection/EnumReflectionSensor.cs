using System;
using UnityEngine;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal class EnumReflectionSensor: ReflectionSensorBase
    {
        Array m_Values;

        internal EnumReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, GetEnumObservationSize(reflectionSensorInfo.GetMemberType()))
        {
            m_Values = Enum.GetValues(reflectionSensorInfo.GetMemberType());
        }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            // Write the enum value as a one-hot encoding.
            // If it's not one of the enum values, put a 1 in the last "bit"
            var enumValue = GetReflectedValue();

            var knownValue = false;
            int i = 0;
            foreach(var val in m_Values)
            {
                if (val.Equals(enumValue))
                {
                    writer[i] = 1.0f;
                    knownValue = true;
                }
                else
                {
                    writer[i] = 0.0f;
                }
                i++;
            }

            writer[i] = knownValue ? 0.0f : 1.0f;
        }

        internal static int GetEnumObservationSize(Type t)
        {
            var values = Enum.GetValues(t);
            // Account for all enum values, plus an extra for unknown values.
            return values.Length + 1;
        }
    }
}
