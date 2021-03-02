using System.Collections.Generic;
using System.Collections.ObjectModel;
using Unity.MLAgents.Sensors;
using UnityEngine;


public class VectorSensorComponent : SensorComponent
{
    int m_observationSize;
    ObservationType m_ObservationType;

    public int ObservationSize
    {
        get { return m_observationSize; }
        set { m_observationSize = value; }
    }

    public VectorSensor sensor;
    public ObservationType ObservationType
    {
        get { return m_ObservationType; }
        set { m_ObservationType = value; }
    }


    /// <summary>
    /// Creates a VectorSensor.
    /// </summary>
    /// <returns></returns>
    public override ISensor CreateSensor()
    {
        sensor = new VectorSensor(m_observationSize, observationType: m_ObservationType);
        return sensor;
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        return new[] { m_observationSize };
    }
}
