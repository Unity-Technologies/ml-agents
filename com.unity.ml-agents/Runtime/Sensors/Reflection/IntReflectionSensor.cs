namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps an integer field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class IntReflectionSensor : ReflectionSensorBase
    {
        public IntReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 1)
        { }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var intVal = (System.Int32)GetReflectedValue();
            writer[0] = (float)intVal;
        }
    }
}
