namespace Unity.MLAgents.Sensors.Reflection
{
    internal class FloatReflectionSensor : ReflectionSensorBase
    {
        internal FloatReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 1)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var floatVal = (System.Single)GetReflectedValue();
            writer[0] = floatVal;
        }
    }
}
