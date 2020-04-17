using UnityEngine;

namespace MLAgents.Sensors
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
        /// The Camera used for rendering the sensor observations.
        /// </summary>
        public Camera camera
        {
            get { return m_Camera; }
            set { m_Camera = value; }
        }

        /// <summary>
        /// The compression type used by the sensor.
        /// </summary>
        public SensorCompressionType compressionType
        {
            get { return m_CompressionType;  }
            set { m_CompressionType = value; }
        }


        /// <summary>
        /// Creates and returns the camera sensor.
        /// </summary>
        /// <param name="camera">Camera object to capture images from.</param>
        /// <param name="width">The width of the generated visual observation.</param>
        /// <param name="height">The height of the generated visual observation.</param>
        /// <param name="grayscale">Whether to convert the generated image to grayscale or keep color.</param>
        /// <param name="name">The name of the camera sensor.</param>
        /// <param name="compression">The compression to apply to the generated image.</param>
        public CameraSensor(
            Camera camera, int width, int height, bool grayscale, string name, SensorCompressionType compression)
        {
            m_Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = GenerateShape(width, height, grayscale);
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
        /// Accessor for the size of the sensor data. Will be h x w x 1 for grayscale and
        /// h x w x 3 for color.
        /// </summary>
        /// <returns>Size of each of the three dimensions.</returns>
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
                DestroyTexture(texture);
                return compressed;
            }
        }

        /// <summary>
        /// Writes out the generated, uncompressed image to the provided <see cref="WriteAdapter"/>.
        /// </summary>
        /// <param name="adapter">Where the observation is written to.</param>
        /// <returns></returns>
        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("CameraSensor.WriteToTensor"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                var numWritten = Utilities.TextureToTensorProxy(texture, adapter, m_Grayscale);
                DestroyTexture(texture);
                return numWritten;
            }
        }

        /// <inheritdoc/>
        public void Update() {}

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public SensorCompressionType GetCompressionType()
        {
            return m_CompressionType;
        }

        /// <summary>
        /// Renders a Camera instance to a 2D texture at the corresponding resolution.
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

        /// <summary>
        /// Computes the observation shape for a camera sensor based on the height, width
        /// and grayscale flag.
        /// </summary>
        /// <param name="width">Width of the image captures from the camera.</param>
        /// <param name="height">Height of the image captures from the camera.</param>
        /// <param name="grayscale">Whether or not to convert the image to grayscale.</param>
        /// <returns>The observation shape.</returns>
        internal static int[] GenerateShape(int width, int height, bool grayscale)
        {
            return new[] { height, width, grayscale ? 1 : 3 };
        }

        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                // TODO move to extension methods for UnityEngine.Object?
                UnityEngine.Object.DestroyImmediate(texture);
            }
            else
            {
                UnityEngine.Object.Destroy(texture);
            }
        }
    }
}
