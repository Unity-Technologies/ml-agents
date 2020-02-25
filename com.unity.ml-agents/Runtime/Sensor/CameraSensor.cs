using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A sensor that wraps a Camera object to generate visual observations for an agent.
    /// </summary>
    public class CameraSensor : ISensor
    {
        Camera m_Camera;
        int m_Width;
        int m_Height;
        bool m_Grayscale;
        string m_Name;
        int[] m_Shape;
        SensorCompressionType m_CompressionType;

        /// <summary>
        /// Creates and returns the camera sensor.
        /// </summary>
        /// <param name="camera">Camera object to capture images from</param>
        /// <param name="width">The width of the generated visual observation</param>
        /// <param name="height">The height of the generated visual observation</param>
        /// <param name="grayscale">Whether to convert the generated image to grayscale or keep color</param>
        /// <param name="name">The name of the camera sensor</param>
        /// <param name="compression">The compression to apply to the generated image</param>
        public CameraSensor(
            Camera camera, int width, int height, bool grayscale, string name, SensorCompressionType compression)
        {
            m_Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, grayscale ? 1 : 3 };
            m_CompressionType = compression;
        }

        /// <summary>
        /// Accessor for the name of the sensor.
        /// </summary>
        /// <returns>Sensor name.</returns>
        public string GetName()
        {
            return m_Name;
        }

        /// <summary>
        /// Accessor for the size of the sensor data.
        /// </summary>
        /// <returns>Size of each dimension. Will be 2D for grayscale and 3D for color.</returns>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <summary>
        /// Generates a compressed image. This can be valuable in speeding-up training.
        /// </summary>
        /// <returns>Compressed image.</returns>
        public byte[] GetCompressedObservation()
        {
            using (TimerStack.Instance.Scoped("CameraSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                UnityEngine.Object.Destroy(texture);
                return compressed;
            }
        }

        /// <summary>
        /// Writes out the generated, uncompressed image to the provided <see cref="WriteAdapter"/>
        /// </summary>
        /// <param name="adapter">Where the observation is written to.</param>
        /// <returns></returns>
        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("CameraSensor.WriteToTensor"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                var numWritten = Utilities.TextureToTensorProxy(texture, adapter, m_Grayscale);
                UnityEngine.Object.Destroy(texture);
                return numWritten;
            }
        }

        /// <inheritdoc/>
        public void Update() {}

        /// <inheritdoc/>
        public SensorCompressionType GetCompressionType()
        {
            return m_CompressionType;
        }

        /// <summary>
        /// Converts a Camera instance and corresponding resolution to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsCamera">Camera.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(Camera obsCamera, int width, int height)
        {
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
            var oldRec = obsCamera.rect;
            obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
            var depth = 24;
            var format = RenderTextureFormat.Default;
            var readWrite = RenderTextureReadWrite.Default;

            var tempRt =
                RenderTexture.GetTemporary(width, height, depth, format, readWrite);

            var prevActiveRt = RenderTexture.active;
            var prevCameraRt = obsCamera.targetTexture;

            // render to offscreen texture (readonly from CPU side)
            RenderTexture.active = tempRt;
            obsCamera.targetTexture = tempRt;

            obsCamera.Render();

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);

            obsCamera.targetTexture = prevCameraRt;
            obsCamera.rect = oldRec;
            RenderTexture.active = prevActiveRt;
            RenderTexture.ReleaseTemporary(tempRt);
            return texture2D;
        }
    }
}
