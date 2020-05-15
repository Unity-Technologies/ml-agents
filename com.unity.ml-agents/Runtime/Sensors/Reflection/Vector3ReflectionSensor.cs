namespace Unity.MLAgents.Sensors.Reflection
{
    /// <summary>
    /// Sensor that wraps a Vector3 field or property of an object, and returns
    /// that as an observation.
    /// </summary>
    internal class Vector3ReflectionSensor : ReflectionSensorBase
    {
        public Vector3ReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 3)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var vecVal = (UnityEngine.Vector3)GetReflectedValue();
            writer[0] = vecVal.x;
            writer[1] = vecVal.y;
            writer[2] = vecVal.z;
        }
    }
}
