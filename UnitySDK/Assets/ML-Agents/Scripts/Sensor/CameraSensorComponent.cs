using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensorComponent : SensorComponent
    {
        public new Camera camera;
        public string sensorName = "CameraSensor";
        public int width = 84;
        public int height = 84;
        public bool grayscale;

        public override ISensor CreateSensor()
        {
            return new CameraSensor(camera, width, height, grayscale, sensorName);
        }

        public override int[] GetObservationShape()
        {
            return new[] { height, width, grayscale ? 1 : 3 };
        }
    }
}
