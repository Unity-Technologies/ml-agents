namespace Unity.MLAgents.Sensors.Reflection
{
    internal class IntReflectionSensor : ReflectionSensorBase
    {
        internal IntReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 1)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var intVal = (System.Int32)GetReflectedValue();
            writer[0] = intVal;
        }
    }
}
