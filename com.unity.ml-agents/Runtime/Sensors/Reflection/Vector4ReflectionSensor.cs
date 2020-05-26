namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a Vector4 field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class Vector4ReflectionSensor : ReflectionSensorBase
    {
        public Vector4ReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 4)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var vecVal = (UnityEngine.Vector4)GetReflectedValue();
            writer[0] = vecVal.x;
            writer[1] = vecVal.y;
            writer[2] = vecVal.z;
            writer[3] = vecVal.w;
        }
    }
}
