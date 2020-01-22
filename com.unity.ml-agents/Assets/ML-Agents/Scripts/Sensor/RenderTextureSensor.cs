using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RenderTextureSensor : ISensor
    {
        RenderTexture m_RenderTexture;
        bool m_Grayscale;
        string m_Name;
        int[] m_Shape;
        SensorCompressionType m_CompressionType;

        public RenderTextureSensor(RenderTexture renderTexture, bool grayscale, string name,
            SensorCompressionType compressionType)
        {
            m_RenderTexture = renderTexture;
            var width = renderTexture != null ? renderTexture.width : 0;
            var height = renderTexture != null ? renderTexture.height : 0;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, grayscale ? 1 : 3 };
            m_CompressionType = compressionType;
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            using(TimerStack.Instance.Scoped("RenderTexSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_RenderTexture);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                UnityEngine.Object.Destroy(texture);
                return compressed;
            }
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("RenderTexSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_RenderTexture);
                var numWritten = Utilities.TextureToTensorProxy(texture, adapter, m_Grayscale);
                UnityEngine.Object.Destroy(texture);
                return numWritten;
            }
        }

        public void Update() { }

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
    }
}
