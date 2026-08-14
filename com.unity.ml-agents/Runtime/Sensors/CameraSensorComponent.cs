using System;
using System.Collections.Generic;
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

        // One sensor per Agent sharing this component
        readonly List<CameraSensor> m_Sensors = new List<CameraSensor>();

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
        /// Whether to generate grayscale images or color.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool Grayscale
        {
            get { return m_Grayscale; }
            set { m_Grayscale = value; }
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

        void Start()
        {
            UpdateSensor();
        }

        /// <summary>
        /// Creates the <see cref="CameraSensor"/>
        /// </summary>
        /// <returns>The created <see cref="CameraSensor"/> object for this component.</returns>
        public override ISensor[] CreateSensors()
        {
            var sensor = new CameraSensor(m_Camera, m_Width, m_Height, Grayscale, m_SensorName, m_Compression, m_ObservationType);
            m_Sensors.Add(sensor);

            if (ObservationStacks != 1)
            {
                return new ISensor[] { new StackingSensor(sensor, ObservationStacks) };
            }
            return new ISensor[] { sensor };
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            foreach (var sensor in m_Sensors)
            {
                sensor.Camera = m_Camera;
                sensor.CompressionType = m_Compression;
                sensor.Camera.enabled = m_RuntimeCameraEnable;
            }
        }

        /// <summary>
        /// Clean up the sensors created by CreateSensors().
        /// </summary>
        public void Dispose()
        {
            foreach (var sensor in m_Sensors)
            {
                sensor.Dispose();
            }
            m_Sensors.Clear();
        }
    }
}
