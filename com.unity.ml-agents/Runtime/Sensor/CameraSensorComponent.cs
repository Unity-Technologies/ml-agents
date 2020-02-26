using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A SensorComponent that creates a <see cref="CameraSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Camera Sensor", (int)MenuGroup.Sensors)]
    public class CameraSensorComponent : SensorComponent
    {
        /// <summary>
        /// Camera object that provides the data to the sensor.
        /// </summary>
        public new Camera camera;

        /// <summary>
        /// Name of the generated <see cref="CameraSensor"/> object.
        /// </summary>
        public string sensorName = "CameraSensor";

        /// <summary>
        /// Width of the generated image.
        /// </summary>
        public int width = 84;

        /// <summary>
        /// Height of the generated image.
        /// </summary>
        public int height = 84;

        /// <summary>
        /// Whether to generate grayscale images or color.
        /// </summary>
        public bool grayscale;

        /// <summary>
        /// The compression type to use for the sensor.
        /// </summary>
        public SensorCompressionType compression = SensorCompressionType.PNG;

        /// <summary>
        /// Creates the <see cref="CameraSensor"/>
        /// </summary>
        /// <returns>The created <see cref="CameraSensor"/> object for this component.</returns>
        public override ISensor CreateSensor()
        {
            return new CameraSensor(camera, width, height, grayscale, sensorName, compression);
        }

        /// <summary>
        /// Computes the observation shape of the sensor.
        /// </summary>
        /// <returns>The observation shape of the associated <see cref="CameraSensor"/> object.</returns>
        public override int[] GetObservationShape()
        {
            return CameraSensor.GenerateShape(width, height, grayscale);
        }
    }
}
