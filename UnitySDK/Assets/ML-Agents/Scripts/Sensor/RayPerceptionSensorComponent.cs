using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RayPerceptionSensorComponent : SensorComponent
    {
        [Tooltip("List of tags in the scene to compare against.")]
        public List<string> detectableTags;

        [Range(0, 50)]
        [Tooltip("Number of rays to the left and right of center.")]
        public int raysPerDirection = 3;

        [Range(0, 180)]
        [Tooltip("Cone size for rays. Using 90 degrees will cast rays to the left and right. Greater than 90 degrees will go backwards")]
        public float maxRayDegrees = 70;

        [Tooltip("Whether to use 3D or 2D raycasts.")]
        public bool is3D = true;
        // TODO start/end offsets
        // TODP sphere radius (or 0 for raycasts)
        // TODO layerMask for raycasts
        // TODO rayLenfgh
        // TODO NAME

        public override ISensor CreateSensor()
        {
            var rayAngles = GetRayAngles(raysPerDirection, maxRayDegrees);
            var name = "raycaster";
            var rayLength = 20.0f;
            return new RayPerceptionSensor(name, transform, rayLength, detectableTags, rayAngles, is3D);
        }

        static float[] GetRayAngles(int raysPerDirection, float maxRayDegrees)
        {
            // Example:
            // { 90, 90 - delta, 90 + delta, 90 - 2*delta, 90 + 2*delta }
            var anglesOut = new float[2 * raysPerDirection + 1];
            var delta = maxRayDegrees / raysPerDirection;
            anglesOut[0] = 90f;
            for (var i = 0; i < raysPerDirection; i++)
            {
                anglesOut[2 * i + 1] = 90 - i * delta;
                anglesOut[2 * i + 2] = 90 + i * delta;
            }
            return anglesOut;
        }
    }
}
