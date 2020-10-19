using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Sensor class that wraps a [RenderTexture](https://docs.unity3d.com/ScriptReference/RenderTexture.html) instance.
    /// </summary>
    public class RenderTextureSensor : ISensor
    {
        RenderTexture m_RenderTexture;
        bool m_Grayscale;
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


        /// <summary>
        /// Initializes the sensor.
        /// </summary>
        /// <param name="renderTexture">The [RenderTexture](https://docs.unity3d.com/ScriptReference/RenderTexture.html)
        /// instance to wrap.</param>
        /// <param name="grayscale">Whether to convert it to grayscale or not.</param>
        /// <param name="name">Name of the sensor.</param>
        /// <param name="compressionType">Compression method for the render texture.</param>
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        public RenderTextureSensor(
            RenderTexture renderTexture, bool grayscale, string name, SensorCompressionType compressionType)
        {
            m_RenderTexture = renderTexture;
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, grayscale ? 1 : 3 };
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
            using (TimerStack.Instance.Scoped("RenderTextureSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_RenderTexture);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                DestroyTexture(texture);
                return compressed;
            }
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("RenderTextureSensor.Write"))
            {
                var texture = ObservationToTexture(m_RenderTexture);
                var numWritten = writer.WriteTexture(texture, m_Grayscale);
                DestroyTexture(texture);
                return numWritten;
            }
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

        /// <summary>
        /// Converts a RenderTexture to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsTexture">RenderTexture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(RenderTexture obsTexture)
        {
            var height = obsTexture.height;
            var width = obsTexture.width;
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

            var prevActiveRt = RenderTexture.active;
            RenderTexture.active = obsTexture;

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);
            texture2D.Apply();
            RenderTexture.active = prevActiveRt;
            return texture2D;
        }

        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                // TODO move to extension methods for UnityEngine.Object?
                Object.DestroyImmediate(texture);
            }
            else
            {
                Object.Destroy(texture);
            }
        }
    }
}
