using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    abstract class SensorComponent : MonoBehaviour
    {
        public abstract ISensor CreateSensor();
    }
}
