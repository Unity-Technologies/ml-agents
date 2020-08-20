namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a float field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class FloatReflectionSensor : ReflectionSensorBase
    {
        public FloatReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 1)
        { }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var floatVal = (System.Single)GetReflectedValue();
            writer[0] = floatVal;
        }
    }
}
