using UnityEngine;
using Unity.MLAgents.Sensors;


public class TestTextureSensorComponent : SensorComponent
{
    TestTextureSensor m_Sensor;

    public Texture2D TestTexture;

    string m_SensorName = "TextureSensor";

    public string SensorName
    {
        get { return m_SensorName; }
        set { m_SensorName = value; }
    }


    public int ObservationStacks = 4;

    public SensorCompressionType CompressionType = SensorCompressionType.PNG;


    /// <inheritdoc/>
    public override ISensor[] CreateSensors()
    {
        m_Sensor = new TestTextureSensor(TestTexture, SensorName, CompressionType);
        if (ObservationStacks != 1)
        {
            return new ISensor[] { new StackingSensor(m_Sensor, ObservationStacks) };
        }
        return new ISensor[] { m_Sensor };
    }
}

