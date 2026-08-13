using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A SensorComponent that creates a <see cref="CameraSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Camera Sensor", (int)MenuGroup.Sensors)]
    public class CameraSensorComponent : SensorComponent, IDisposable
    {
        [HideInInspector, SerializeField, FormerlySerializedAs("camera")]
        Camera m_Camera;

        CameraSensor m_Sensor;

        /// <summary>
        /// Camera object that provides the data to the sensor.
        /// </summary>
        public Camera Camera
        {
            get { return m_Camera; }
            set { m_Camera = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("sensorName")]
        string m_SensorName = "CameraSensor";

        /// <summary>
        /// Name of the generated <see cref="CameraSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("width")]
        int m_Width = 84;

        /// <summary>
        /// Width of the generated observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int Width
        {
            get { return m_Width; }
            set { m_Width = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("height")]
        int m_Height = 84;

        /// <summary>
        /// Height of the generated observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int Height
        {
            get { return m_Height; }
            set { m_Height = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("grayscale")]
        bool m_Grayscale;

        /// <summary>
        /// Whether to generate grayscale images or color. Disable RGBD to use it.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool Grayscale
        {
            get { return m_Grayscale; }
            set { m_Grayscale = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("rgbd")]
        bool m_RGBD;

        /// <summary>
        /// Whether to generate color+depth images. RGBD has priority over Grayscale.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool RGBD
        {
            get { return m_RGBD; }
            set { m_RGBD = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField]
        ObservationType m_ObservationType;

        /// <summary>
        /// The type of the observation.
        /// </summary>
        public ObservationType ObservationType
        {
            get { return m_ObservationType; }
            set { m_ObservationType = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField]
        bool m_RuntimeCameraEnable;


        /// <summary>
        /// Controls the whether the camera sensor's attached camera
        /// is enabled during runtime. Overrides the camera object enabled status.
        /// Disabled for improved performance. Disabled by default.
        /// </summary>
        public bool RuntimeCameraEnable
        {
            get { return m_RuntimeCameraEnable; }
            set { m_RuntimeCameraEnable = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField]
        [Range(1, 50)]
        [Tooltip("Number of camera frames that will be stacked before being fed to the neural network.")]
        int m_ObservationStacks = 1;

        [HideInInspector, SerializeField, FormerlySerializedAs("compression")]
        SensorCompressionType m_Compression = SensorCompressionType.PNG;

        /// <summary>
        /// The compression type to use for the sensor.
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

        /// <summary>
        /// The material used to render the depth image.
        /// </summary>
        private Material m_DepthMaterial;

        void Start()
        {
            UpdateSensor();
            m_DepthMaterial = new Material(Shader.Find("Custom/DepthShader"));
        }

        /// <summary>
        /// Creates the <see cref="CameraSensor"/>
        /// </summary>
        /// <returns>The created <see cref="CameraSensor"/> object for this component.</returns>
        public override ISensor[] CreateSensors()
        {
            Dispose();
            m_Sensor = new CameraSensor(m_Camera, m_Width, m_Height, Grayscale, RGBD, m_SensorName, m_Compression, m_ObservationType);

            if (ObservationStacks != 1)
            {
                return new ISensor[] { new StackingSensor(m_Sensor, ObservationStacks) };
            }
            return new ISensor[] { m_Sensor };
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                // Update depth settings before camera settings because m_Compression might change
                if (m_RGBD)
                {
                    m_Grayscale = false;
                    m_Compression = SensorCompressionType.OPENEXR;
                }

                // Update camera settings
                m_Sensor.Camera = m_Camera;
                m_Sensor.CompressionType = m_Compression;
                m_Sensor.Camera.enabled = m_RuntimeCameraEnable;
            }
        }

        /// <summary>
        /// Clean up the sensor created by CreateSensors().
        /// </summary>
        public void Dispose()
        {
            if (!ReferenceEquals(m_Sensor, null))
            {
                m_Sensor.Dispose();
                m_Sensor = null;
            }
        }

        /// <summary>
        /// Apply the depth material to the camera image if the sensor is set to RGBD.
        /// </summary>
        void OnRenderImage(RenderTexture src, RenderTexture dest)
        {
            if (m_RGBD && m_Sensor != null && m_Sensor.m_InCameraSensorRender)
            {
                Graphics.Blit(src, dest, m_DepthMaterial);
            }
            else
            {
                Graphics.Blit(src, dest);
            }
        }
    }
}
