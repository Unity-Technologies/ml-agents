using UnityEngine;
using Unity.MLAgents.Sensors;

public class TestTextureSensor : ISensor
{
    Texture2D m_Texture;
    string m_Name;
    private ObservationSpec m_ObservationSpec;
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
        m_ObservationSpec = ObservationSpec.Visual(3, height, width);
        m_CompressionType = compressionType;
    }

    /// <inheritdoc/>
    public string GetName()
    {
        return m_Name;
    }

    /// <inheritdoc/>
    public ObservationSpec GetObservationSpec()
    {
        return m_ObservationSpec;
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
    public CompressionSpec GetCompressionSpec()
    {
        return CompressionSpec.Default();
    }
}

