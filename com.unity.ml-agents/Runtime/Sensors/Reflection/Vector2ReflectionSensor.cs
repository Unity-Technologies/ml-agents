namespace Unity.MLAgents.Sensors.Reflection
{
    internal class Vector2ReflectionSensor : ReflectionSensorBase
    {
        internal Vector2ReflectionSensor(ReflectionSensorInfo reflectionSensorInfo)
            : base(reflectionSensorInfo, 2)
        {}

        internal override void WriteReflectedField(ObservationWriter writer)
        {
            var vecVal = (UnityEngine.Vector2)GetReflectedValue();
            writer[0] = vecVal.x;
            writer[1] = vecVal.y;
        }
    }
}
