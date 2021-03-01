using System.Collections.Generic;
using System.Collections.ObjectModel;
using Unity.MLAgents.Sensors;
using UnityEngine;


public class VectorSensorComponent : SensorComponent
{
    public int observationSize;
    public VectorSensor sensor;
    public ObservationType observationType;

    /// <summary>
    /// Creates a VectorSensor.
    /// </summary>
    /// <returns></returns>
    public override ISensor CreateSensor()
    {
        sensor = new VectorSensor(observationSize, observationType: observationType);
        return sensor;
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        return new[] { observationSize };
    }
}
