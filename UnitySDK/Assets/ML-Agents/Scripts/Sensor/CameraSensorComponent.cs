using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    class CameraSensorComponent : SensorComponent
    {
        public new Camera camera;
        public int width = 84;
        public int height = 84;
        public bool grayscale = false;

        public override ISensor CreateSensor()
        {
            return new CameraSensor(camera, width, height, grayscale);
        }
    }
}
