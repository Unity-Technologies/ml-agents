using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    [AddComponentMenu("ML Agents/Render Texture Sensor", (int) MenuGroup.Sensors)]
    public class RenderTextureSensorComponent : SensorComponent
    {
        public RenderTexture renderTexture;
        public string sensorName = "RenderTextureSensor";
        public bool grayscale;
        public SensorCompressionType compression = SensorCompressionType.PNG;

        public override ISensor CreateSensor()
        {
            return new RenderTextureSensor(renderTexture, grayscale, sensorName, compression);
        }

        public override int[] GetObservationShape()
        {
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;

            return new[] { height, width, grayscale ? 1 : 3 };
        }
    }
}
