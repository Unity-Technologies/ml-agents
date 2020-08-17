namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a Vector2 field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class Vector2ReflectionSensor : ReflectionSensorBase
    {
        public Vector2ReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 2)
        { }

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var vecVal = (UnityEngine.Vector2)GetReflectedValue();
            writer[0] = vecVal.x;
            writer[1] = vecVal.y;
        }
    }
}
