using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RenderTextureSensorComponent : SensorComponent
    {
        public RenderTexture renderTexture;
        public string sensorName = "RenderTextureSensor";
        public bool grayscale;

        public override ISensor CreateSensor()
        {
            return new RenderTextureSensor(renderTexture, grayscale, sensorName);
        }

        public override int[] GetObservationShape()
        {
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;

            return new[] { height, width, grayscale ? 1 : 3 };
        }
    }
}
