using MLAgents;
namespace MLAgentsExamples
{
    public class NoopVisualSensorComponent : SensorComponent
    {
        public string sensorName = "NoopSensor";
        public int width = 84;
        public int height = 84;
        public bool grayscale;

        public override ISensor CreateSensor()
        {
            return new NoopVisualSensor(width, height, grayscale, sensorName);
        }

        public override int[] GetObservationShape()
        {
            return new[] { height, width, grayscale ? 1 : 3 };
        }
    }
}
