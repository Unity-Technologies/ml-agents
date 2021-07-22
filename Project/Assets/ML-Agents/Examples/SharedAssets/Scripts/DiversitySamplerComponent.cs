using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

public class DiversitySamplerComponent : SensorComponent
{
    public string SensorName = "DiversitySampler";
    public bool ContinuousSampling;
    public bool BalancedSampling;
    public bool DisableSampling;
    public int DiversitySize;
    public int DiscreteSetting;
    [Range(-1f, 1f)]
    public float[] ContinuousSetting;

    DiversitySampler m_Sensor;

    /// <summary>
    /// Creates a DiversitySampler
    /// </summary>
    /// <returns></returns>
    public override ISensor[] CreateSensors()
    {
        m_Sensor = new DiversitySampler(SensorName, DiversitySize, ContinuousSampling, BalancedSampling);
        OnValidate();

        return new ISensor[] { m_Sensor };
    }

    // These settings need to change on the sensor at runtime
    public void OnValidate()
    {
        if (m_Sensor != null)
        {
            m_Sensor.DisableSampling = DisableSampling;
            if (DisableSampling)
            {
                m_Sensor.DiscreteSetting = DiscreteSetting;
                m_Sensor.ContinuousSetting = ContinuousSetting;
            }
        }
    }
}
