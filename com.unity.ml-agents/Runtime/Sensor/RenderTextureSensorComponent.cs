using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Component that wraps a <see cref="RenderTextureSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Render Texture Sensor", (int)MenuGroup.Sensors)]
    public class RenderTextureSensorComponent : SensorComponent
    {
        /// <summary>
        /// The <see cref="RenderTexture"/> instance that the associated
        /// <see cref="RenderTextureSensor"/> wraps.
        /// </summary>
        public RenderTexture renderTexture;

        /// <summary>
        /// Name of the sensor.
        /// </summary>
        public string sensorName = "RenderTextureSensor";

        /// <summary>
        /// Whether the RenderTexture observation should be converted to grayscale or not.
        /// </summary>
        public bool grayscale;

        /// <summary>
        /// Compression type for the render texture observation.
        /// </summary>
        public SensorCompressionType compression = SensorCompressionType.PNG;

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            return new RenderTextureSensor(renderTexture, grayscale, sensorName, compression);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;

            return new[] { height, width, grayscale ? 1 : 3 };
        }
    }
}
