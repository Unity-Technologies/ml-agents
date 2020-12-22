using System;

namespace Unity.MLAgents.Sensors.Reflection
{
    internal class EnumReflectionSensor : ReflectionSensorBase
    {
        Array m_Values;
        bool m_IsFlags;

        internal EnumReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, GetEnumObservationSize(reflectionSensorInfo.GetMemberType()))
        {
            var memberType = reflectionSensorInfo.GetMemberType();
            m_Values = Enum.GetValues(memberType);
            m_IsFlags = memberType.IsDefined(typeof(FlagsAttribute), false);
        }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            // Write the enum value as a one-hot encoding.
            // Note that unknown enum values will record all 0's.
            // Flags will get treated as a sequence of bools.
            var enumValue = (Enum)GetReflectedValue();

            int i = 0;
            foreach (var val in m_Values)
            {
                if (m_IsFlags)
                {
                    if (enumValue.HasFlag((Enum)val))
                    {
                        writer[i] = 1.0f;
                    }
                    else
                    {
                        writer[i] = 0.0f;
                    }
                }
                else
                {
                    if (val.Equals(enumValue))
                    {
                        writer[i] = 1.0f;
                    }
                    else
                    {
                        writer[i] = 0.0f;
                    }
                }
                i++;
            }
        }

        internal static int GetEnumObservationSize(Type t)
        {
            var values = Enum.GetValues(t);
            // Account for all enum values
            return values.Length;
        }
    }
}
