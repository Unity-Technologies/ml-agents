using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensor : ISensor
    {
        Camera m_Camera;
        int m_Width;
        int m_Height;
        bool m_Grayscale;
        string m_Name;
        int[] m_Shape;
        SensorCompressionType m_CompressionType;

        public CameraSensor(Camera camera, int width, int height, bool grayscale, string name,
            SensorCompressionType compression)
        {
            m_Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, grayscale ? 1 : 3 };
            m_CompressionType = compression;
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
            using (TimerStack.Instance.Scoped("CameraSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                UnityEngine.Object.Destroy(texture);
                return compressed;
            }
        }

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

        public void Update() { }

        public SensorCompressionType GetCompressionType()
        {
            return m_CompressionType;
        }

        /// <summary>
        /// Converts a m_Camera and corresponding resolution to a 2D texture.
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
