using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RenderTextureSensorComponent : SensorComponent
    {
        public RenderTexture renderTexture;
        public string sensorName = "RenderTextureSensor";
        public int width = 84;
        public int height = 84;
        public bool grayscale = false;

        public override ISensor CreateSensor()
        {
            return new RenderTextureSensor(renderTexture, width, height, grayscale, sensorName);
        }
    }
}
