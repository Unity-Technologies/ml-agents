using UnityEngine;
using Unity.MLAgents.Sensors;

public class TestTextureSensor : ISensor
{
    Texture2D m_Texture;
    string m_Name;
    int[] m_Shape;
    SensorCompressionType m_CompressionType;

    /// <summary>
    /// The compression type used by the sensor.
    /// </summary>
    public SensorCompressionType CompressionType
    {
        get { return m_CompressionType; }
        set { m_CompressionType = value; }
    }


    public TestTextureSensor(
        Texture2D texture, string name, SensorCompressionType compressionType)
    {
        m_Texture = texture;
        var width = texture.width;
        var height = texture.height;
        m_Name = name;
        m_Shape = new[] { height, width, 3 };
        m_CompressionType = compressionType;
    }

    /// <inheritdoc/>
    public string GetName()
    {
        return m_Name;
    }

    /// <inheritdoc/>
    public int[] GetObservationShape()
    {
        return m_Shape;
    }

    /// <inheritdoc/>
    public byte[] GetCompressedObservation()
    {
        var compressed = m_Texture.EncodeToPNG();
        return compressed;
    }

    /// <inheritdoc/>
    public int Write(ObservationWriter writer)
    {
        var numWritten = writer.WriteTexture(m_Texture, false);
        return numWritten;
    }

    /// <inheritdoc/>
    public void Update() { }

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public SensorCompressionType GetCompressionType()
    {
        return m_CompressionType;
    }
}

