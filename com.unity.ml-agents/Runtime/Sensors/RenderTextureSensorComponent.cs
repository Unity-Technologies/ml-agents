using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents.Sensors
{
    /// <summary>
    /// Component that wraps a <see cref="RenderTextureSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Render Texture Sensor", (int)MenuGroup.Sensors)]
    public class RenderTextureSensorComponent : SensorComponent
    {
        RenderTextureSensor m_Sensor;

        /// <summary>
        /// The <see cref="RenderTexture"/> instance that the associated
        /// <see cref="RenderTextureSensor"/> wraps.
        /// </summary>
        [HideInInspector, SerializeField, FormerlySerializedAs("renderTexture")]
        RenderTexture m_RenderTexture;

        public RenderTexture renderTexture
        {
            get { return m_RenderTexture;  }
            set { m_RenderTexture = value;  }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("sensorName")]
        string m_SensorName = "RenderTextureSensor";

        /// <summary>
        /// Name of the generated <see cref="RenderTextureSensor"/>.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string sensorName
        {
            get { return m_SensorName;  }
            set { m_SensorName = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("grayscale")]
        public bool m_Grayscale;

        /// <summary>
        /// Whether the RenderTexture observation should be converted to grayscale or not.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool grayscale
        {
            get { return m_Grayscale;  }
            set { m_Grayscale = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("compression")]
        SensorCompressionType m_Compression = SensorCompressionType.PNG;

        /// <summary>
        /// Compression type for the render texture observation.
        /// </summary>
        public SensorCompressionType compression
        {
            get { return m_Compression;  }
            set { m_Compression = value; UpdateSensor(); }
        }

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            m_Sensor = new RenderTextureSensor(renderTexture, grayscale, sensorName, compression);
            return m_Sensor;
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;

            return new[] { height, width, grayscale ? 1 : 3 };
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                m_Sensor.compressionType = m_Compression;
            }
        }
    }
}
