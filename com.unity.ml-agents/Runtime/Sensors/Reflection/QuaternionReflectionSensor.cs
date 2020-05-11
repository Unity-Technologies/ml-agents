namespace Unity.MLAgents.Sensors.Reflection
{
    internal class QuaternionReflectionSensor : ReflectionSensorBase
    {
        internal QuaternionReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
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
