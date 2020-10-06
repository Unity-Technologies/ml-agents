using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Component that wraps a <see cref="RenderTextureSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Render Texture Sensor", (int)MenuGroup.Sensors)]
    public class RenderTextureSensorComponent : SensorComponent
    {
        RenderTextureSensor m_Sensor;

        /// <summary>
        /// The [RenderTexture](https://docs.unity3d.com/ScriptReference/RenderTexture.html) instance
        /// that the associated <see cref="RenderTextureSensor"/> wraps.
        /// </summary>
        [HideInInspector, SerializeField, FormerlySerializedAs("renderTexture")]
        RenderTexture m_RenderTexture;

        /// <summary>
        /// Stores the [RenderTexture](https://docs.unity3d.com/ScriptReference/RenderTexture.html)
        /// associated with this sensor.
        /// </summary>
        public RenderTexture RenderTexture
        {
            get { return m_RenderTexture; }
            set { m_RenderTexture = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("sensorName")]
        string m_SensorName = "RenderTextureSensor";

        /// <summary>
        /// Name of the generated <see cref="RenderTextureSensor"/>.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("grayscale")]
        bool m_Grayscale;

        /// <summary>
        /// Whether the RenderTexture observation should be converted to grayscale or not.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool Grayscale
        {
            get { return m_Grayscale; }
            set { m_Grayscale = value; }
        }

        [HideInInspector, SerializeField]
        [Range(1, 50)]
        [Tooltip("Number of frames that will be stacked before being fed to the neural network.")]
        int m_ObservationStacks = 1;

        [HideInInspector, SerializeField, FormerlySerializedAs("compression")]
        SensorCompressionType m_Compression = SensorCompressionType.PNG;

        /// <summary>
        /// Compression type for the render texture observation.
        /// </summary>
        public SensorCompressionType CompressionType
        {
            get { return m_Compression; }
            set { m_Compression = value; UpdateSensor(); }
        }

        /// <summary>
        /// Whether to stack previous observations. Using 1 means no previous observations.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int ObservationStacks
        {
            get { return m_ObservationStacks; }
            set { m_ObservationStacks = value; }
        }

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            m_Sensor = new RenderTextureSensor(RenderTexture, Grayscale, SensorName, m_Compression);
            if (ObservationStacks != 1)
            {
                return new StackingSensor(m_Sensor, ObservationStacks);
            }
            return m_Sensor;
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            var width = RenderTexture != null ? RenderTexture.width : 0;
            var height = RenderTexture != null ? RenderTexture.height : 0;
            var observationShape = new[] { height, width, Grayscale ? 1 : 3 };

            var stacks = ObservationStacks > 1 ? ObservationStacks : 1;
            if (stacks > 1)
            {
                observationShape[2] *= stacks;
            }

            return observationShape;
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                m_Sensor.CompressionType = m_Compression;
            }
        }
    }
}
