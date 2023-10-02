namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a boolean field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class BoolReflectionSensor : ReflectionSensorBase
    {
        public BoolReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 1)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var boolVal = (System.Boolean)GetReflectedValue();
            writer[0] = boolVal ? 1.0f : 0.0f;
        }
    }
}
