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
    public override ISensor CreateSensor()
    {
        m_Sensor = new TestTextureSensor(TestTexture, SensorName, CompressionType);
        if (ObservationStacks != 1)
        {
            return new StackingSensor(m_Sensor, ObservationStacks);
        }
        return m_Sensor;
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        var width = TestTexture.width;
        var height = TestTexture.height;
        var observationShape = new[] { height, width, 3 };

        var stacks = ObservationStacks > 1 ? ObservationStacks : 1;
        if (stacks > 1)
        {
            observationShape[2] *= stacks;
        }

        return observationShape;
    }
}

