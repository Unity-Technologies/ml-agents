using System;
using System.Threading;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    class RenderTextureSensor : ISensor
    {
        private RenderTexture m_RenderTexture;
        private int m_Width;
        private int m_Height;
        private bool m_Grayscale;
        private string m_Name;
        private int[] m_Shape;

        public RenderTextureSensor(RenderTexture renderTexture, int width, int height, bool grayscale, string name)
        {
            m_RenderTexture = renderTexture;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { width, height, grayscale ? 1 : 3 };
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetFloatObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            using(TimerStack.Instance.Scoped("RenderTexSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_RenderTexture, m_Width, m_Height);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                UnityEngine.Object.Destroy(texture);
                return compressed;
            }
        }

        public void WriteToTensor(TensorProxy tensorProxy, int index)
        {
            using (TimerStack.Instance.Scoped("RenderTexSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_RenderTexture, m_Width, m_Height);
                Utilities.TextureToTensorProxy(texture, tensorProxy, m_Grayscale, index);
                UnityEngine.Object.Destroy(texture);
            }
        }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.PNG;
        }

        /// <summary>
        /// Converts a RenderTexture and correspinding resolution to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsTexture">RenderTexture.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(RenderTexture obsTexture, int width, int height)
        {
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

            if (width != texture2D.width || height != texture2D.height)
            {
                texture2D.Resize(width, height);
            }

            if (width != obsTexture.width || height != obsTexture.height)
            {
                throw new UnityAgentsException(string.Format(
                    "RenderTexture {0} : width/height is {1}/{2} brain is expecting {3}/{4}.",
                    obsTexture.name, obsTexture.width, obsTexture.height, width, height));
            }

            var prevActiveRt = RenderTexture.active;
            RenderTexture.active = obsTexture;

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);
            texture2D.Apply();
            RenderTexture.active = prevActiveRt;
            return texture2D;
        }
    }
}
