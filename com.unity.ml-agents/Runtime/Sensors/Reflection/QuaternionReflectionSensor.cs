namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a quaternion field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class QuaternionReflectionSensor : ReflectionSensorBase
    {
        public QuaternionReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 4)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var quatVal = (UnityEngine.Quaternion)GetReflectedValue();
            writer[0] = quatVal.x;
            writer[1] = quatVal.y;
            writer[2] = quatVal.z;
            writer[3] = quatVal.w;
        }
    }
}
