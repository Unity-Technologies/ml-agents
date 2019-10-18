using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    class RenderTextureSensorComponent : SensorComponent
    {
        public RenderTexture renderTexture;
        public int width = 84;
        public int height = 84;
        public bool grayscale = false;

        public override ISensor CreateSensor()
        {
            return new RenderTextureSensor(renderTexture, width, height, grayscale);
        }
    }
}
